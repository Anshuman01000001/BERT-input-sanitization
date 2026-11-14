"""FastAPI application for the game chat moderation service.

Endpoints:

    POST /moderate      moderate one chat message (requires X-API-Key)
    GET  /health        liveness and readiness
    GET  /demo          a small page for live demonstrations
    GET  /demo/config   demo page settings

The game server calls ``/moderate`` once per chat message and applies the
returned action:

    allow        deliver the message
    flag         deliver the message and queue it for human review
    block        reject the message and tell the sender why
    shadow_mute  deliver the message to the sender only

Authentication and the request-body limit live in :mod:`app.security`, which
runs as raw ASGI middleware ahead of routing: FastAPI resolves a request body
before it runs route dependencies, so a dependency-level limit would let an
unauthenticated caller make the service buffer an arbitrarily large body.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader
from redis.exceptions import RedisError

from app import __version__
from app.config import Settings, get_settings
from app.harassment import HarassmentTracker
from app.logging_config import setup_logging
from app.model import build_model
from app.ratelimit import ApiRateLimiter
from app.redis_client import create_client
from app.schemas import HealthResponse, ModerateRequest, ModerateResponse
from app.scoring import BehaviorScorer
from app.security import SecurityMiddleware
from app.service import ModerationService
from app.spam import SpamChecker

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Paths that require the API key. Kept as a module constant so the security
# middleware and the OpenAPI description cannot drift apart.
PROTECTED_PATHS = ("/moderate",)

# Declared here only so the OpenAPI schema documents the header; the actual
# check happens in app.security, ahead of routing.
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and wire up the pipeline before serving traffic."""
    settings = get_settings()
    setup_logging(settings.log_level)

    problem = settings.api_key_problem()
    if problem:
        raise RuntimeError(f"refusing to start: {problem}")
    if settings.using_default_api_key:
        logger.warning(
            "running with the well-known development API key; "
            "set MOD_ALLOW_INSECURE_DEFAULT_KEY=false before exposing this"
        )

    redis = create_client(settings)
    model = build_model(settings)
    try:
        model.load()
        model.warmup()
    except Exception:
        # A missing artifact, malformed metadata and an ONNX Runtime failure
        # all mean the same thing here: the service cannot do its job.
        logger.exception("the model could not be loaded; refusing to start")
        await redis.aclose()
        raise

    app.state.settings = settings
    app.state.redis = redis
    app.state.model = model
    app.state.rate_limiter = ApiRateLimiter(redis, settings)
    app.state.service = ModerationService(
        settings=settings,
        redis=redis,
        model=model,
        spam=SpamChecker(redis, settings),
        harassment=HarassmentTracker(redis, settings),
        scorer=BehaviorScorer(redis, settings),
    )
    logger.info("moderation service ready", extra={"model_id": model.model_id})

    try:
        yield
    finally:
        await redis.aclose()


app = FastAPI(
    title="Game chat moderation API",
    version=__version__,
    description=__doc__,
    lifespan=lifespan,
)

# Outermost middleware: it bounds the body and checks the API key before
# routing, so neither the body nor the JSON parser is reachable by an
# unauthenticated caller. The settings provider is a lambda so that the
# settings are resolved per request rather than captured at import time.
app.add_middleware(
    SecurityMiddleware,
    settings_provider=lambda: get_settings(),
    protected_paths=PROTECTED_PATHS,
)


def client_identity(request: Request) -> str:
    """Identify the caller for rate limiting.

    The socket peer is used rather than ``X-Forwarded-For``, which a client
    can forge. Behind a proxy every caller then shares one bucket, which the
    global ceiling in :class:`~app.ratelimit.ApiRateLimiter` covers.
    """
    client = request.client
    return client.host if client else "unknown"


async def enforce_api_rate_limit(request: Request) -> None:
    """Limit how fast each client can call the API."""
    limiter: ApiRateLimiter = request.app.state.rate_limiter
    if not await limiter.allow(client_identity(request)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Render validation errors without echoing unencodable input.

    The default handler serializes with ``ensure_ascii=False``, so an error
    that quotes the offending value cannot be encoded when that value is not
    valid UTF-8 - a lone surrogate, for instance. The request would then fail
    with a 500 instead of the 422 it deserves.
    """
    payload = {"detail": jsonable_encoder(exc.errors())}
    return Response(
        content=json.dumps(payload, ensure_ascii=True),
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        media_type="application/json",
    )


@app.post(
    "/moderate",
    response_model=ModerateResponse,
    dependencies=[Security(API_KEY_HEADER), Depends(enforce_api_rate_limit)],
    summary="Moderate a single chat message",
)
async def moderate(payload: ModerateRequest, request: Request) -> ModerateResponse:
    """Decide what should happen to one chat message."""
    settings: Settings = request.app.state.settings
    if len(payload.text) > settings.max_message_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"message is longer than {settings.max_message_chars} characters",
        )
    service: ModerationService = request.app.state.service
    return await service.moderate(payload)


@app.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
async def health(request: Request) -> HealthResponse:
    """Report whether the model is loaded and Redis is reachable."""
    model = request.app.state.model
    redis = request.app.state.redis
    redis_ok = True
    try:
        await redis.ping()
    except RedisError:
        redis_ok = False
    overall = "ok" if model.is_loaded and redis_ok else "degraded"
    return HealthResponse(
        status=overall,
        model_loaded=model.is_loaded,
        model_id=model.model_id,
        redis_ok=redis_ok,
        version=__version__,
    )


@app.get("/demo/config", summary="Configuration for the demo page")
async def demo_config(request: Request) -> dict[str, object]:
    """Settings the demo page needs. Never exposes the API key."""
    if not request.app.state.settings.enable_demo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="not found"
        )
    model = request.app.state.model
    return {
        "model_id": model.model_id,
        "labels": model.labels,
    }


@app.get("/demo", include_in_schema=False, summary="Live demo page")
async def demo_page(request: Request) -> FileResponse:
    """A small page for demonstrating the service."""
    if not request.app.state.settings.enable_demo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="not found"
        )
    return FileResponse(STATIC_DIR / "demo.html")
