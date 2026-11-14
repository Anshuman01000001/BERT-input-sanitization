"""The moderation pipeline.

    Request -> 1. Sanitize -> 2. Spam rules -> 3. Harassment tracker
            -> 4. BERT -> 5. Decision + player score

Cheap checks run first so that most abuse is caught before the model uses any
CPU. When a cheap check already decides the outcome, the classifier is
skipped entirely (``Settings.skip_model_on_spam``).

If Redis is unavailable the service does not fail: the stateful heuristics
(rate limit, duplicates, harassment counters, behaviour score) are skipped,
the classifier still runs, and the response says so. Losing the stateful
layer degrades spam and harassment detection; it must never take the whole
service down.
"""

from __future__ import annotations

import logging
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings
from app.harassment import HarassmentTracker, HarassmentVerdict, ReportVerdict
from app.model import ToxicityModel
from app.normalize import NormalizedMessage, normalize
from app.schemas import ModerateRequest, ModerateResponse
from app.scoring import BehaviorScorer, Escalation
from app.spam import SpamChecker, SpamVerdict

logger = logging.getLogger(__name__)

# Severity ordering, lowest to highest.
_SEVERITY = {"allow": 0, "flag": 1, "shadow_mute": 2, "block": 3}


def worst(*actions: str) -> str:
    """The most severe of the given actions."""
    return max(actions, key=lambda action: _SEVERITY.get(action, 0))


async def _no_harassment() -> HarassmentVerdict:
    """A stand-in for the harassment check when there is nothing to check."""
    return HarassmentVerdict()


async def _no_spam() -> SpamVerdict:
    """A stand-in for the spam check when there is nothing to check."""
    return SpamVerdict()


class ModerationService:
    """Runs the moderation pipeline for a single chat message."""

    def __init__(
        self,
        settings: Settings,
        redis: Redis,
        model: ToxicityModel,
        spam: SpamChecker,
        harassment: HarassmentTracker,
        scorer: BehaviorScorer,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._model = model
        self._spam = spam
        self._harassment = harassment
        self._scorer = scorer

    async def moderate(self, request: ModerateRequest) -> ModerateResponse:
        """Moderate one message and return the decision."""
        started = time.perf_counter()
        settings = self._settings

        reasons: list[str] = []
        action = "allow"
        points = 0
        scores: dict[str, float] = {}
        state_available = True

        # --- 1. Sanitize ---------------------------------------------------
        message = normalize(request.text)

        # --- 2. Spam rules -------------------------------------------------
        # A report is not chat: it must not be counted as a duplicate or a
        # burst, or a victim filing several reports would be muted for it.
        spam_verdict, ok = await self._stateful(
            self._spam.check(
                request.player_id, message, request.account_age_s
            )
            if not request.is_report
            else _no_spam(),
            SpamVerdict(),
        )
        state_available = state_available and ok
        if spam_verdict.reasons:
            reasons.extend(spam_verdict.reasons)
            action = worst(action, spam_verdict.action)
            points += self._spam_points(spam_verdict)

        # --- 3. Harassment tracker ------------------------------------------
        # A report is a meta-action about a player, not a message aimed at
        # them, so it does not create an exchange between the two.
        harassment_verdict, ok = await self._stateful(
            self._harassment.check(request.player_id, request.target_id)
            if not request.is_report
            else _no_harassment(),
            HarassmentVerdict(),
        )
        state_available = state_available and ok
        if harassment_verdict.reasons:
            reasons.extend(harassment_verdict.reasons)
            action = worst(action, harassment_verdict.action)
            points += harassment_verdict.points

        # --- 4. BERT --------------------------------------------------------
        # The classifier is only skipped when the cheap checks have already
        # prevented delivery. A "flag" verdict still lets the message reach
        # other players, so it has to be classified: a threat that also
        # carried a link must still be blocked.
        if action == "shadow_mute" and settings.skip_model_on_spam:
            reasons.append("model_skipped")
        else:
            try:
                scores = self._model.score(message.normalized)
            except Exception:
                logger.exception("the classifier failed; using the cheap checks only")
                scores = {}
                reasons.append("model_unavailable")
            toxic_action, toxic_reasons, toxic_points = self._toxicity_decision(scores)
            reasons.extend(toxic_reasons)
            action = worst(action, toxic_action)
            points += toxic_points
            if request.target_id and scores and not request.is_report:
                await self._stateful(
                    self._harassment.record_toxicity(
                        request.player_id,
                        request.target_id,
                        max(scores.values()),
                        harassment_verdict.entry_id,
                    ),
                    None,
                )

        # --- 5. Decision + behaviour score ----------------------------------
        player_score = await self._read_or_apply_score(
            request.player_id, points, state_available
        )
        escalation = Escalation("none")
        if state_available:
            try:
                escalation = await self._scorer.escalation_for(
                    request.player_id, player_score
                )
            except RedisError:
                logger.warning("could not read the escalation state", exc_info=True)
                state_available = False

        action, reasons = self._apply_escalation(action, reasons, escalation)
        if escalation.level == "mute" and state_available:
            try:
                await self._scorer.note_mute(
                    request.player_id, escalation.mute_seconds or 0
                )
            except RedisError:
                logger.warning("could not record the mute tier", exc_info=True)
                state_available = False

        # --- Player reports -------------------------------------------------
        target_score, report_reason = await self._handle_report(
            request, player_score, state_available
        )
        if report_reason:
            reasons.append(report_reason)

        if not state_available:
            reasons.append("state_unavailable")

        latency_ms = (time.perf_counter() - started) * 1000
        response = ModerateResponse(
            action=action,
            reasons=reasons,
            scores={label: round(value, 4) for label, value in scores.items()},
            player_score=player_score,
            target_score=target_score,
            escalation=escalation.level,
            muted_for_s=(
                escalation.mute_seconds if escalation.level == "mute" else None
            ),
            latency_ms=round(latency_ms, 2),
        )
        self._log_decision(request, message, response)
        return response

    # --- pieces of the pipeline -------------------------------------------

    def _spam_points(self, verdict: SpamVerdict) -> int:
        """Points for spam signals, following the handoff's scoring table."""
        settings = self._settings
        points = 0
        if "rate_limit" in verdict.reasons:
            points += settings.points_rate_limit
        other_signals = [reason for reason in verdict.reasons if reason != "rate_limit"]
        if other_signals:
            points += settings.points_spam
        return points

    def _toxicity_decision(
        self, scores: dict[str, float]
    ) -> tuple[str, list[str], int]:
        """Apply the per-category thresholds to the classifier output.

        Hard categories (threat, identity hate, severe toxicity) block at a
        low threshold; soft categories (toxic, insult, obscene) only flag at a
        high one, because gaming banter is common and mostly harmless.
        """
        settings = self._settings
        reasons: list[str] = []
        blocked = False
        flagged = False

        for label, score in scores.items():
            if label in settings.hard_label_set and score >= settings.block_threshold:
                reasons.append(f"{label}:{score:.2f}")
                blocked = True
            elif label in settings.soft_label_set and score >= settings.flag_threshold:
                reasons.append(f"{label}:{score:.2f}")
                flagged = True

        if blocked:
            return "block", reasons, settings.points_toxic_block
        if flagged:
            return "flag", reasons, settings.points_toxic_flag
        return "allow", reasons, 0

    def _apply_escalation(
        self, action: str, reasons: list[str], escalation: Escalation
    ) -> tuple[str, list[str]]:
        """Let the behaviour score raise the action up the ladder."""
        if escalation.level == "mute":
            reasons.append(f"escalation:mute:{escalation.mute_seconds}s")
            return worst(action, "shadow_mute"), reasons
        if escalation.level == "ban_review":
            reasons.append("escalation:ban_review")
            return worst(action, "flag"), reasons
        if escalation.level == "flag":
            reasons.append("escalation:flag")
            return worst(action, "flag"), reasons
        if escalation.level == "warn":
            reasons.append("escalation:warn")
        return action, reasons

    async def _read_or_apply_score(
        self, player_id: str, points: int, state_available: bool
    ) -> int:
        """Apply the points for this message and return the new score."""
        if not state_available:
            return 0
        try:
            if points:
                return await self._scorer.apply(player_id, points)
            return await self._scorer.get(player_id)
        except RedisError:
            logger.warning("could not update the behaviour score", exc_info=True)
            return 0

    async def _handle_report(
        self, request: ModerateRequest, player_score: int, state_available: bool
    ) -> tuple[int | None, str | None]:
        """Register a player report against ``target_id``.

        The report counts against the *target*, weighted by the reporter's own
        history so that a serial false-reporter loses influence.

        Returns the target's new score and the reason code, if any.
        """
        if not request.is_report or not request.target_id:
            return None, None
        if not state_available:
            return None, None

        target_score = 0
        try:
            target_score = await self._scorer.get(request.target_id)
        except RedisError:
            logger.warning("could not read the target's score", exc_info=True)
            return None, None

        report, _ = await self._stateful(
            self._harassment.register_report(
                request.player_id,
                request.target_id,
                reporter_score=player_score,
            ),
            ReportVerdict(0, "report_unavailable", 0.0, 0),
        )
        if report.reason == "report_unavailable":
            return target_score, None

        try:
            if report.points:
                target_score = await self._scorer.apply(
                    request.target_id, report.points
                )
        except RedisError:
            logger.warning("could not apply the report points", exc_info=True)
        return target_score, report.reason

    # --- infrastructure ----------------------------------------------------

    async def _stateful(self, coro, default):
        """Run a Redis-backed operation, degrading to ``default`` if Redis is down."""
        try:
            return await coro, True
        except RedisError:
            logger.warning(
                "redis unavailable; stateful moderation checks are disabled",
                exc_info=True,
            )
            return default, False

    def _log_decision(
        self,
        request: ModerateRequest,
        message: NormalizedMessage,
        response: ModerateResponse,
    ) -> None:
        """Record the decision for moderator review and player appeals."""
        decision: dict[str, object] = {
            "player_id": request.player_id,
            "target_id": request.target_id,
            "channel": request.channel,
            "normalized_hash": message.message_hash,
            "action": response.action,
            "reasons": response.reasons,
            "scores": response.scores,
            "player_score": response.player_score,
            "escalation": response.escalation,
            "latency_ms": response.latency_ms,
        }
        if self._settings.log_message_text:
            decision["text"] = message.original
        logger.info("moderation decision", extra={"decision": decision})
