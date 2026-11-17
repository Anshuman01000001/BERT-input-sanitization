"""Tests for the harassment tracker."""

from __future__ import annotations

import pytest

from app.harassment import HarassmentTracker, is_escalating, mean


@pytest.fixture
def tracker(settings, redis_client) -> HarassmentTracker:
    return HarassmentTracker(redis_client, settings)


class TestHelpers:
    def test_mean_of_empty_list_is_zero(self):
        assert mean([]) == 0.0

    def test_mean(self):
        assert mean([0.0, 1.0]) == 0.5

    def test_escalation_needs_enough_history(self):
        assert is_escalating([0.9, 0.9], 0.15) is False

    def test_escalation_detects_a_rising_trend(self):
        assert is_escalating([0.9, 0.9, 0.1, 0.1], 0.15) is True

    def test_escalation_ignores_a_flat_trend(self):
        assert is_escalating([0.5, 0.5, 0.5, 0.5], 0.15) is False

    def test_escalation_ignores_a_falling_trend(self):
        assert is_escalating([0.1, 0.1, 0.9, 0.9], 0.15) is False


class TestUnansweredTargeting:
    async def test_no_signal_for_a_single_message(self, tracker):
        verdict = await tracker.check("sender", "target")
        assert "unanswered_targeting" not in verdict.reasons

    async def test_detects_one_sided_contact(self, tracker):
        verdict = None
        for _ in range(4):
            verdict = await tracker.check("sender", "target")
        assert "unanswered_targeting" in verdict.reasons
        assert verdict.action == "shadow_mute"
        assert verdict.points == 10

    async def test_replies_make_the_contact_two_sided(self, tracker):
        for _ in range(4):
            await tracker.check("sender", "target")
        for _ in range(3):
            await tracker.check("target", "sender")
        verdict = await tracker.check("sender", "target")
        assert "unanswered_targeting" not in verdict.reasons

    async def test_a_message_to_yourself_is_ignored(self, tracker):
        verdict = await tracker.check("me", "me")
        assert verdict.reasons == ()
        assert verdict.action == "allow"


class TestPileOn:
    async def test_detects_many_senders_against_one_target(self, tracker):
        await tracker.check("s1", "victim")
        await tracker.check("s2", "victim")
        verdict = await tracker.check("s3", "victim")
        assert "pile_on" in verdict.reasons
        assert verdict.action == "flag"

    async def test_the_same_sender_is_not_a_pile_on(self, tracker):
        verdict = None
        for _ in range(3):
            verdict = await tracker.check("s1", "victim")
        assert "pile_on" not in verdict.reasons

    async def test_a_pile_on_does_not_blame_one_participant(self, tracker):
        await tracker.check("s1", "victim")
        await tracker.check("s2", "victim")
        verdict = await tracker.check("s3", "victim")
        assert verdict.action == "flag"
        assert verdict.points == 0


class TestEscalation:
    async def test_detects_rising_toxicity_in_a_pair(self, tracker):
        verdict = await tracker.check("sender", "target")
        await tracker.record_toxicity("sender", "target", 0.1, verdict.entry_id)
        verdict = await tracker.check("sender", "target")
        await tracker.record_toxicity("sender", "target", 0.2, verdict.entry_id)
        verdict = await tracker.check("sender", "target")
        await tracker.record_toxicity("sender", "target", 0.95, verdict.entry_id)
        verdict = await tracker.check("sender", "target")
        assert "escalating_toxicity" in verdict.reasons

    async def test_a_flat_history_is_not_escalation(self, tracker):
        for _ in range(4):
            verdict = await tracker.check("sender", "target")
            await tracker.record_toxicity("sender", "target", 0.5, verdict.entry_id)
        verdict = await tracker.check("sender", "target")
        assert "escalating_toxicity" not in verdict.reasons

    async def test_the_score_lands_on_the_right_entry(self, tracker):
        """Two messages in flight: each score must reach its own entry."""
        first = await tracker.check("sender", "target")
        second = await tracker.check("sender", "target")
        await tracker.record_toxicity("sender", "target", 0.99, second.entry_id)
        history = await tracker._pair_history("sender", "target")
        newest = history[0]
        assert newest.entry_id == second.entry_id
        assert newest.toxicity == 0.99
        assert first.entry_id != second.entry_id


class TestReports:
    """A report is a claim, so it only counts when something corroborates it."""

    @staticmethod
    async def make_exchange(tracker, toxicity: float) -> None:
        """The target sends the reporter one message, scored ``toxicity``."""
        verdict = await tracker.check("target", "reporter")
        await tracker.record_toxicity("target", "reporter", toxicity, verdict.entry_id)

    async def test_a_report_without_an_exchange_counts_for_nothing(self, tracker):
        report = await tracker.register_report("reporter", "stranger", 0)
        assert report.points == 0
        assert report.reason == "report_no_history"

    async def test_a_report_against_benign_behaviour_moves_nothing(self, tracker):
        await self.make_exchange(tracker, 0.05)
        report = await tracker.register_report("reporter", "target", 0)
        assert report.points == 0
        assert report.reason == "report"

    async def test_a_mild_message_does_not_corroborate_a_report(self, tracker):
        """The evidence bar is separate from the action band.

        A message that is merely held back from other players is not strong
        evidence that a target abused someone, and a report carries ten
        points - so corroboration has its own threshold rather than reusing
        the one at which the service starts hiding messages.
        """
        await self.make_exchange(tracker, 0.67)
        report = await tracker.register_report("reporter", "target", 0)
        assert report.points == 0
        assert report.reason == "report"

    async def test_a_report_is_confirmed_by_the_targets_own_toxicity(self, tracker):
        await self.make_exchange(tracker, 0.95)
        report = await tracker.register_report("reporter", "target", 0)
        assert report.reason == "confirmed_report"
        assert report.points == 10

    async def test_confirmation_does_not_depend_on_the_targets_score(self, tracker):
        await self.make_exchange(tracker, 0.95)
        report = await tracker.register_report("reporter", "target", 0)
        assert report.reason == "confirmed_report"

    async def test_self_reports_are_ignored(self, tracker):
        report = await tracker.register_report("me", "me", 0)
        assert report.points == 0
        assert report.reason == "self_report_ignored"

    async def test_serial_reporters_lose_trust(self, tracker):
        await self.make_exchange(tracker, 0.95)
        for _ in range(4):
            await tracker.register_report("spammer", "target", 0)
        report = await tracker.register_report("spammer", "target", 0)
        assert report.weight < 1.0
        assert report.points < 10

    async def test_reporters_with_a_bad_score_are_discounted(self, tracker):
        await self.make_exchange(tracker, 0.95)
        report = await tracker.register_report("toxic_reporter", "target", 20)
        assert report.weight < 1.0
        assert report.points < 10

    async def test_one_reporter_cannot_drive_a_target_up_the_ladder(self, tracker):
        await self.make_exchange(tracker, 0.95)
        total = 0
        for _ in range(20):
            total += (await tracker.register_report("attacker", "target", 0)).points
        assert total <= 10
