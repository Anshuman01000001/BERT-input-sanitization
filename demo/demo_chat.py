"""Scripted demonstration of the moderation pipeline.

Run it in-process (no server, no Redis needed):

    .venv/bin/python demo/demo_chat.py

Or point it at a running instance:

    .venv/bin/python demo/demo_chat.py --server http://localhost:8000 --api-key secret

Each scenario prints the decision, the reasons behind it and the classifier
scores, which is what the game server would act on.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The settings resolve `models/` and `.env` relative to the working
# directory, so run from the repository root whatever the caller's cwd is.
os.chdir(REPO_ROOT)

from app.config import Settings  # noqa: E402
from app.harassment import HarassmentTracker  # noqa: E402
from app.model import build_model  # noqa: E402
from app.schemas import ModerateRequest, ModerateResponse  # noqa: E402
from app.scoring import BehaviorScorer  # noqa: E402
from app.service import ModerationService  # noqa: E402
from app.spam import SpamChecker  # noqa: E402

ACTION_LABEL = {
    "allow": "ALLOW",
    "flag": "FLAG",
    "block": "BLOCK",
    "shadow_mute": "SHADOW MUTE",
}


@dataclass
class Scenario:
    """A named sequence of messages that demonstrates one behaviour."""

    name: str
    note: str
    requests: list[ModerateRequest] = field(default_factory=list)


def build_scenarios() -> list[Scenario]:
    """The demo script: clean, toxic, threat, spam, harassment, evasion."""
    return [
        Scenario(
            name="Clean chat",
            note="ordinary banter, nothing to act on",
            requests=[
                ModerateRequest(text="gg, nice shot!", player_id="alice"),
                ModerateRequest(text="nice heal, thanks!", player_id="bob"),
            ],
        ),
        Scenario(
            name="Toxic message",
            note="a soft category at 0.97: flagged for review, not blocked",
            requests=[
                ModerateRequest(text="you are a disgusting idiot", player_id="carol"),
            ],
        ),
        Scenario(
            name="Threatening message",
            note="a hard category at 0.84: blocked outright",
            requests=[
                ModerateRequest(
                    text="I will find you and break your legs", player_id="dave"
                ),
            ],
        ),
        Scenario(
            name="Spam burst",
            note="a link plus a burst of messages: shadow-muted",
            requests=[
                ModerateRequest(
                    text="join my discord discord.gg/abcd1234", player_id="eve"
                ),
                ModerateRequest(text="free skins click here now", player_id="eve"),
                ModerateRequest(text="free skins click here now", player_id="eve"),
                ModerateRequest(text="free skins click here now", player_id="eve"),
                ModerateRequest(text="free skins click here now", player_id="eve"),
                ModerateRequest(text="free skins click here now", player_id="eve"),
            ],
        ),
        Scenario(
            name="Harassment",
            note="four unanswered messages aimed at one player",
            requests=[
                ModerateRequest(
                    text="why do you keep ignoring me",
                    player_id="mallory",
                    target_id="victim",
                ),
                ModerateRequest(
                    text="answer me right now please",
                    player_id="mallory",
                    target_id="victim",
                ),
                ModerateRequest(
                    text="i can see you online",
                    player_id="mallory",
                    target_id="victim",
                ),
                ModerateRequest(
                    text="you really are rude today",
                    player_id="mallory",
                    target_id="victim",
                ),
            ],
        ),
        Scenario(
            name="Evasion attempt",
            note="leet speak and Cyrillic lookalikes, normalized before scoring",
            requests=[
                ModerateRequest(text="n00b f@gg0t sk1ll", player_id="trudy"),
            ],
        ),
        Scenario(
            name="Player report",
            note=(
                "a report is recorded but only counts when the target's own "
                "messages corroborate it, so it adds nothing here"
            ),
            requests=[
                ModerateRequest(
                    text="reporting this behaviour",
                    player_id="victim",
                    target_id="mallory",
                    is_report=True,
                ),
            ],
        ),
    ]


class InProcessRunner:
    """Runs the pipeline in-process against an in-memory Redis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis = None
        self._service: ModerationService | None = None
        self.model_id = "unknown"

    async def __aenter__(self) -> InProcessRunner:
        import fakeredis.aioredis as fakeredis

        self._redis = fakeredis.FakeRedis(decode_responses=True)
        model = build_model(self._settings)
        model.load()
        self.model_id = model.model_id or "unknown"
        self._service = ModerationService(
            settings=self._settings,
            redis=self._redis,
            model=model,
            spam=SpamChecker(self._redis, self._settings),
            harassment=HarassmentTracker(self._redis, self._settings),
            scorer=BehaviorScorer(self._redis, self._settings),
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def send(self, request: ModerateRequest) -> ModerateResponse:
        assert self._service is not None
        return await self._service.moderate(request)


class HttpRunner:
    """Sends the same messages to a running instance over HTTP."""

    def __init__(self, url: str, api_key: str) -> None:
        self._url = url.rstrip("/")
        self._api_key = api_key
        self.model_id = "unknown"

    async def __aenter__(self) -> HttpRunner:
        import httpx

        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        async with httpx.AsyncClient(base_url=self._url, headers=headers, timeout=10) as client:
            response = await client.get("/demo/config")
            response.raise_for_status()
            self.model_id = response.json().get("model_id", "unknown")
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    async def send(self, request: ModerateRequest) -> ModerateResponse:
        import httpx

        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        payload = request.model_dump(exclude_none=True)
        async with httpx.AsyncClient(
            base_url=self._url, headers=headers, timeout=10
        ) as client:
            response = await client.post("/moderate", json=payload)
            response.raise_for_status()
            return ModerateResponse.model_validate(response.json())


def print_response(request: ModerateRequest, response: ModerateResponse) -> None:
    """Render one decision."""
    target = f" -> {request.target_id}" if request.target_id else ""
    print(f"    {request.player_id}{target}: {request.text!r}")
    print(f"      -> {ACTION_LABEL.get(response.action, response.action.upper())}")

    if response.reasons:
        print(f"         reasons: {', '.join(response.reasons)}")
    if response.scores:
        top = sorted(response.scores.items(), key=lambda item: item[1], reverse=True)[:3]
        scores = "  ".join(f"{label}={value:.2f}" for label, value in top if value >= 0.05)
        if scores:
            print(f"         scores:  {scores}")
    print(
        f"         score:   {response.player_score}   "
        f"escalation: {response.escalation}   {response.latency_ms} ms"
    )
    if response.target_score is not None:
        print(f"         target score: {response.target_score}")
    print()


async def run(runner, scenarios: list[Scenario]) -> None:
    for scenario in scenarios:
        print(f"\n=== {scenario.name} ===")
        print(f"    ({scenario.note})")
        for request in scenario.requests:
            response = await runner.send(request)
            print_response(request, response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        help="Base URL of a running instance; omit to run in-process.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MOD_API_KEY", ""),
        help="API key for --server; defaults to $MOD_API_KEY.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = Settings()
    scenarios = build_scenarios()

    runner = (
        HttpRunner(args.server, args.api_key)
        if args.server
        else InProcessRunner(settings)
    )

    async with runner:
        print("Game chat moderation - scripted demonstration")
        print(f"model: {runner.model_id}")
        await run(runner, scenarios)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
