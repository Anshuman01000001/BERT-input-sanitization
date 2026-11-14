"""Tests for the behaviour score and the escalation ladder."""

from __future__ import annotations

import pytest

from app.redis_client import mute_key, now_seconds, score_key
from app.scoring import BehaviorScorer


@pytest.fixture
def scorer(settings, redis_client) -> BehaviorScorer:
    return BehaviorScorer(redis_client, settings)


class TestScore:
    async def test_a_new_player_has_no_score(self, scorer):
        assert await scorer.get("p1") == 0

    async def test_points_are_added(self, scorer):
        assert await scorer.apply("p2", 5) == 5

    async def test_points_accumulate(self, scorer):
        await scorer.apply("p3", 5)
        assert await scorer.apply("p3", 3) == 8

    async def test_the_score_never_goes_negative(self, scorer):
        assert await scorer.apply("p4", -10) == 0


class TestDecay:
    async def test_the_score_decays_over_time(self, scorer, redis_client):
        await scorer.apply("p5", 10)
        # pretend the score was last updated five hours ago
        await redis_client.hset(score_key("p5"), "updated_at", now_seconds() - 5 * 3600)
        assert await scorer.get("p5") == 5

    async def test_applying_points_also_applies_the_decay(self, scorer, redis_client):
        await scorer.apply("p6", 10)
        await redis_client.hset(score_key("p6"), "updated_at", now_seconds() - 5 * 3600)
        assert await scorer.apply("p6", 2) == 7


class TestEscalation:
    async def test_a_clean_player_is_at_the_bottom(self, scorer):
        escalation = await scorer.escalation_for("p7", 0)
        assert escalation.level == "none"

    async def test_repeat_offenders_are_warned(self, scorer):
        escalation = await scorer.escalation_for("p8", 5)
        assert escalation.level == "warn"

    async def test_persistent_offenders_are_flagged(self, scorer):
        escalation = await scorer.escalation_for("p9", 15)
        assert escalation.level == "flag"

    async def test_severe_offenders_are_muted(self, scorer):
        escalation = await scorer.escalation_for("p10", 30)
        assert escalation.level == "mute"
        assert escalation.mute_seconds == 600

    async def test_the_worst_offenders_go_to_human_review(self, scorer):
        escalation = await scorer.escalation_for("p11", 60)
        assert escalation.level == "ban_review"


class TestMuteTiers:
    async def test_the_mute_length_is_kept_for_the_whole_incident(self, scorer):
        first = await scorer.escalation_for("p12", 30)
        await scorer.note_mute("p12", first.mute_seconds)
        second = await scorer.escalation_for("p12", 30)
        assert second.mute_seconds == 600

    async def test_a_new_incident_gets_a_longer_mute(self, scorer):
        first = await scorer.escalation_for("p13", 30)
        await scorer.note_mute("p13", first.mute_seconds)
        # dropping back below the band ends the incident
        await scorer.escalation_for("p13", 0)
        second = await scorer.escalation_for("p13", 30)
        assert first.mute_seconds == 600
        assert second.mute_seconds == 3600

    async def test_the_mute_counter_advances_once_per_incident(self, scorer, redis_client):
        escalation = await scorer.escalation_for("p14", 30)
        await scorer.note_mute("p14", escalation.mute_seconds)
        await scorer.note_mute("p14", escalation.mute_seconds)
        assert await redis_client.get(mute_key("p14")) == "1"
