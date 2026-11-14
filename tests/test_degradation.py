"""Tests for the degraded paths.

Redis going away must not take the service down: the stateful heuristics are
skipped, the classifier still runs, and the response says what was lost.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.harassment import HarassmentTracker
from app.schemas import ModerateRequest
from app.scoring import BehaviorScorer
from app.service import ModerationService
from app.spam import SpamChecker


class BrokenRedis:
    """A Redis client whose every command fails, as if the server were down."""

    def pipeline(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def hgetall(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def get(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def incr(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def expire(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def lrange(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def lpush(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def lindex(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")

    async def lset(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")


class StubModel:
    """A classifier that returns fixed probabilities."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.is_loaded = True
        self.model_id = "stub"
        self.labels = list(scores)

    def score(self, text: str) -> dict[str, float]:
        return dict(self._scores)

    def score_batch(self, texts: list[str]) -> list[dict[str, float]]:
        return [dict(self._scores) for _ in texts]


def build_service(settings, redis, model) -> ModerationService:
    return ModerationService(
        settings=settings,
        redis=redis,
        model=model,
        spam=SpamChecker(redis, settings),
        harassment=HarassmentTracker(redis, settings),
        scorer=BehaviorScorer(redis, settings),
    )


@pytest.fixture
def threat_scores() -> dict[str, float]:
    return {
        "toxic": 0.85,
        "severe_toxic": 0.09,
        "obscene": 0.05,
        "threat": 0.82,
        "insult": 0.08,
        "identity_hate": 0.04,
    }


class TestRedisUnavailable:
    async def test_the_classifier_still_decides(self, settings, threat_scores):
        service = build_service(settings, BrokenRedis(), StubModel(threat_scores))
        response = await service.moderate(
            ModerateRequest(text="I will find you", player_id="p1")
        )
        assert response.action == "block"
        assert any(reason.startswith("threat:") for reason in response.reasons)

    async def test_the_response_says_the_state_was_lost(self, settings, threat_scores):
        service = build_service(settings, BrokenRedis(), StubModel(threat_scores))
        response = await service.moderate(
            ModerateRequest(text="hello there", player_id="p1")
        )
        assert "state_unavailable" in response.reasons

    async def test_no_score_is_reported_when_the_state_is_gone(
        self, settings, threat_scores
    ):
        service = build_service(settings, BrokenRedis(), StubModel(threat_scores))
        response = await service.moderate(
            ModerateRequest(text="hello there", player_id="p1")
        )
        assert response.player_score == 0
        assert response.escalation == "none"

    async def test_a_clean_message_is_still_allowed(self, settings):
        scores = dict.fromkeys(("toxic", "threat", "insult"), 0.01)
        service = build_service(settings, BrokenRedis(), StubModel(scores))
        response = await service.moderate(
            ModerateRequest(text="gg, nice shot", player_id="p1")
        )
        assert response.action == "allow"


class TestReportWithoutState:
    async def test_a_report_without_redis_is_not_counted(self, settings, threat_scores):
        service = build_service(settings, BrokenRedis(), StubModel(threat_scores))
        response = await service.moderate(
            ModerateRequest(
                text="reporting this",
                player_id="reporter",
                target_id="culprit",
                is_report=True,
            )
        )
        assert response.target_score is None
        assert "state_unavailable" in response.reasons
