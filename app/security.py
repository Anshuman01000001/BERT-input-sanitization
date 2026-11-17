"""Request-level security: authentication and body-size limits.

This is raw ASGI middleware on purpose. FastAPI resolves a request body
*before* it runs route dependencies, so a FastAPI-level size check would let
an unauthenticated caller make the service buffer an arbitrarily large body
before the API key was ever looked at.

Running here, ahead of routing, means:

* the body is bounded while it is still streaming in, so a chunked request
  with no ``Content-Length`` cannot exceed the limit either;
* the API key is checked before any body is read, before any JSON is parsed,
  and before the classifier can be reached.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings
from app.ratelimit import LocalRateLimiter


class SecurityMiddleware:
    """Authenticates protected paths and bounds the request body."""

    def __init__(
        self,
        app: ASGIApp,
        settings_provider: Callable[[], Settings],
        protected_paths: tuple[str, ...],
        body_limit_provider: Callable[[str, Settings], int] | None = None,
    ) -> None:
        self._app = app
        self._settings_provider = settings_provider
        self._protected_paths = protected_paths
        # Optional per-path body limit. The default cap is sized for a chat
        # message; a route that legitimately carries whole files needs its
        # own, and the cap still applies before any body is read.
        self._body_limit_provider = body_limit_provider
        # Failed authentication attempts are counted here, locally: the
        # Redis-backed limiter only sees authenticated traffic, so without
        # this a brute-force attempt against the key would be unmetered.
        self._failed_auth = LocalRateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        settings = self._settings_provider()
        headers = Headers(scope=scope)

        if self._is_protected(scope.get("path", "")) and settings.require_api_key:
            provided = headers.getlist("x-api-key")
            if len(provided) > 1:
                # A proxy that picks a different duplicate than we do would
                # disagree about which key was presented. Refuse instead.
                await self._send_json(
                    send, 400, {"detail": "more than one X-API-Key header"}
                )
                return
            candidate = provided[0] if provided else None
            if not self._key_matches(candidate, settings.api_key):
                if not self._failed_auth.allow(
                    self._client_id(scope),
                    settings.failed_auth_limit,
                    settings.failed_auth_window_s,
                ):
                    await self._send_json(
                        send,
                        429,
                        {"detail": "too many failed authentication attempts"},
                    )
                    return
                await self._send_json(
                    send,
                    401,
                    {
                        "detail": (
                            "invalid or missing X-API-Key header. Send the same "
                            "value as the service's MOD_API_KEY, or start the "
                            "service with MOD_REQUIRE_API_KEY=false for local testing."
                        )
                    },
                )
                return

        if scope.get("method") in ("POST", "PUT", "PATCH"):
            limit = settings.max_body_bytes
            if self._body_limit_provider is not None:
                limit = self._body_limit_provider(scope.get("path", ""), settings)
            body, too_large = await self._read_body(receive, limit)
            if too_large:
                await self._send_json(
                    send, 413, {"detail": "request body too large"}
                )
                return
            await self._app(scope, self._replay(body), send)
            return

        await self._app(scope, receive, send)

    def _is_protected(self, path: str) -> bool:
        """Whether a path needs the API key.

        A protected path protects its sub-paths too, so a route like
        ``/compress/{handle}`` cannot be reached by dropping the prefix.
        """
        return any(
            path == protected or path.startswith(f"{protected}/")
            for protected in self._protected_paths
        )

    @staticmethod
    def _key_matches(candidate: str | None, expected: str) -> bool:
        """Constant-time comparison, safe for any non-ASCII input."""
        if candidate is None:
            return False
        return hmac.compare_digest(
            candidate.encode("utf-8"), expected.encode("utf-8")
        )

    @staticmethod
    def _client_id(scope: Scope) -> str:
        """The socket peer, used to meter failed authentication attempts."""
        client = scope.get("client")
        if isinstance(client, tuple) and client:
            return str(client[0])
        return "unknown"

    async def _read_body(
        self, receive: Receive, max_bytes: int
    ) -> tuple[bytes, bool]:
        """Read the body, giving up as soon as it exceeds ``max_bytes``."""
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > max_bytes:
                return b"", True
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        return b"".join(chunks), False

    @staticmethod
    def _replay(body: bytes) -> Receive:
        """Hand the already-read body to the application."""
        sent = False

        async def receive() -> Message:
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        return receive

    @staticmethod
    async def _send_json(send: Send, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
