"""Harassment tracking between players.

Three independent signals, because no single one is reliable on its own:

* **Unanswered targeting** - one player keeps sending at another who never
  replies. Repeated, one-sided contact is the classic harassment shape.
* **Escalation** - the toxicity of the recent messages in a pair is rising.
  The classifier score of earlier messages is stored alongside them, so the
  trend can be seen before it is acted on.
* **Pile-on** - many *different* players target the same player in a short
  window. This is a property of the target, not of any one sender, so it is
  flagged for human review rather than punished against a participant.

Player reports are a fourth signal, weighted by the reporter's own history so
that a griefing account cannot flood a target with false reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.config import Settings
from app.redis_client import (
    now_seconds,
    pair_key,
    pair_messages_key,
    pileon_key,
    report_points_key,
    reports_against_key,
    reports_made_key,
)

logger = logging.getLogger(__name__)

# How many times to retry a WATCH-protected update before giving up.
MAX_WATCH_RETRIES = 8


@dataclass(frozen=True, slots=True)
class PairMessage:
    """One message in a sender/target conversation history."""

    ts: int
    sender: str
    toxicity: float
    entry_id: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "from": self.sender,
                "toxic": self.toxicity,
                "id": self.entry_id,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> PairMessage | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or "from" not in parsed:
            return None
        try:
            return cls(
                ts=int(parsed.get("ts", 0)),
                sender=str(parsed["from"]),
                toxicity=float(parsed.get("toxic", 0.0)),
                entry_id=str(parsed.get("id", "")),
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class HarassmentVerdict:
    """Outcome of the harassment checks.

    ``action`` is ``allow``, ``flag`` or ``shadow_mute``. ``points`` is what
    the sender's behaviour score should increase by.
    """

    action: str = "allow"
    reasons: tuple[str, ...] = ()
    points: int = 0
    detail: dict[str, float] = field(default_factory=dict)
    # Identifies this message's history entry, so the classifier score can be
    # written back to the right one when several are in flight.
    entry_id: str = ""

    @property
    def is_harassment(self) -> bool:
        return self.action != "allow"


@dataclass(frozen=True, slots=True)
class ReportVerdict:
    """Outcome of registering a player report against another player."""

    points: int
    reason: str
    weight: float
    reports_against: int


def mean(values: list[float]) -> float:
    """Arithmetic mean of ``values`` (0.0 for an empty list)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def is_escalating(recent_toxicity: list[float], margin: float) -> bool:
    """True when the two most recent scores exceed the older ones by ``margin``.

    ``recent_toxicity`` is ordered newest first.
    """
    if len(recent_toxicity) < 3:
        return False
    newest = recent_toxicity[:2]
    older = recent_toxicity[2:4]
    if not older:
        return False
    return mean(newest) > mean(older) + margin


class HarassmentTracker:
    """Tracks sender-to-target relationships in Redis."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings

    async def check(self, sender_id: str, target_id: str) -> HarassmentVerdict:
        """Evaluate the sender-to-target relationship for one new message."""
        if not target_id or target_id == sender_id:
            return HarassmentVerdict()

        settings = self._settings
        reasons: list[str] = []
        detail: dict[str, float] = {}
        points = 0
        action = "allow"

        pair_count = await self._increment_pair(sender_id, target_id)
        history, entry_id = await self._record_message(sender_id, target_id)

        # Only messages from inside the window count. The list's TTL slides
        # with activity, so without this a pair that talks once a day would
        # still look like an hour-long exchange.
        cutoff = now_seconds() - settings.pair_window_s
        history = [entry for entry in history if entry.ts > cutoff]

        from_sender = sum(1 for entry in history if entry.sender == sender_id)
        from_target = sum(1 for entry in history if entry.sender == target_id)
        if (
            from_sender >= settings.unanswered_threshold
            and from_sender > from_target * 2
        ):
            reasons.append("unanswered_targeting")
            detail["unanswered_count"] = float(from_sender)
            detail["target_replies"] = float(from_target)
            points += settings.points_harassment
            action = "shadow_mute"

        recent_toxicity = [entry.toxicity for entry in history[1:]]
        if is_escalating(recent_toxicity, settings.escalation_margin):
            reasons.append("escalating_toxicity")
            detail["toxicity_trend"] = mean(recent_toxicity[:2])
            points += settings.points_harassment
            action = "shadow_mute"

        senders = await self._register_pileon(sender_id, target_id)
        if senders >= settings.pileon_threshold:
            reasons.append("pile_on")
            detail["pileon_size"] = float(senders)
            # The target is being pile-oned; do not punish one participant.
            if action == "allow":
                action = "flag"

        detail["pair_messages"] = float(pair_count)
        return HarassmentVerdict(
            action=action,
            reasons=tuple(reasons),
            points=points,
            detail=detail,
            entry_id=entry_id,
        )

    async def record_toxicity(
        self, sender_id: str, target_id: str, toxicity: float, entry_id: str
    ) -> None:
        """Store the classifier score of the message that was just checked.

        The pipeline runs the harassment check *before* the classifier (cheap
        checks first), so the history entry is written with a placeholder
        toxicity and patched afterwards. The entry is matched by id rather
        than by position: with several messages from the same pair in flight
        the newest entry is not necessarily the one being scored.
        """
        if not target_id or target_id == sender_id or not entry_id:
            return
        key = pair_messages_key(sender_id, target_id)
        raw_entries = await self._redis.lrange(
            key, 0, self._settings.pair_message_limit - 1
        )
        for index, raw in enumerate(raw_entries):
            entry = PairMessage.from_json(raw)
            if entry is not None and entry.entry_id == entry_id:
                await self._redis.lset(
                    key,
                    index,
                    PairMessage(entry.ts, entry.sender, toxicity, entry.entry_id).to_json(),
                )
                return

    async def register_report(
        self,
        reporter_id: str,
        target_id: str,
        reporter_score: int,
    ) -> ReportVerdict:
        """Record a report and decide how much it counts for.

        A report is a *claim*, not evidence, so it only counts when something
        corroborates it:

        * there must be a real exchange between the two players, so a report
          cannot be aimed at a stranger;
        * the target's own messages in that exchange must have been scored as
          toxic, so a report about harmless behaviour counts for almost
          nothing;
        * the reporter's own history scales the weight, so a serial
          false-reporter loses influence;
        * one reporter may only contribute a bounded amount to one target per
          window, so no single account can drive a player up the ladder.

        Confirmation deliberately does *not* depend on the target's score:
        a score the reporter can move is not evidence.
        """
        settings = self._settings
        if not target_id or reporter_id == target_id:
            return ReportVerdict(
                points=0, reason="self_report_ignored", weight=0.0, reports_against=0
            )

        history = await self._pair_history(reporter_id, target_id)
        if not history:
            return ReportVerdict(
                points=0, reason="report_no_history", weight=0.0, reports_against=0
            )

        target_toxicity = max(
            (
                message.toxicity
                for message in history
                if message.sender == target_id
            ),
            default=0.0,
        )
        corroborated = target_toxicity >= settings.flag_threshold

        made = await self._increment(reports_made_key(reporter_id), settings.score_ttl_s)
        against = await self._increment(
            reports_against_key(target_id), settings.score_ttl_s
        )

        if made <= settings.reporter_trust_full_count:
            trust = 1.0
        elif made <= settings.reporter_trust_half_count:
            trust = 0.5
        else:
            trust = 0.25
        if reporter_score >= settings.score_flag_at:
            trust *= 0.5

        # A claim with nothing behind it moves nothing. Reports about mild
        # behaviour are still recorded and visible to a human reviewer; they
        # just cannot move a player up the ladder on their own, because that
        # is how a report becomes a weapon.
        points = int(settings.report_points * trust) if corroborated else 0

        points = await self._cap_points(reporter_id, target_id, points)
        return ReportVerdict(
            points=points,
            reason="confirmed_report" if corroborated else "report",
            weight=trust,
            reports_against=against,
        )

    async def _cap_points(
        self, reporter_id: str, target_id: str, points: int
    ) -> int:
        """Clamp how much one reporter may add to one target per window.

        The read and the write are guarded by a WATCH so that two concurrent
        reports from the same reporter cannot both see the same remaining
        allowance and together exceed the cap.
        """
        settings = self._settings
        key = report_points_key(reporter_id, target_id)
        for _ in range(MAX_WATCH_RETRIES):
            async with self._redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    spent = int(await pipe.get(key) or 0)
                    allowed = max(0, settings.report_points_cap - spent)
                    allowed_points = min(points, allowed)
                    if not allowed_points:
                        await pipe.unwatch()
                        return 0
                    pipe.multi()
                    pipe.incrby(key, allowed_points)
                    pipe.expire(key, settings.report_points_window_s)
                    await pipe.execute()
                    return allowed_points
                except WatchError:
                    continue
        logger.warning("could not apply the report point cap for %s", reporter_id)
        return 0

    async def _pair_history(
        self, player_a: str, player_b: str
    ) -> list[PairMessage]:
        """The messages exchanged between two players, newest first."""
        key = pair_messages_key(player_a, player_b)
        raw_entries = await self._redis.lrange(
            key, 0, self._settings.pair_message_limit - 1
        )
        history: list[PairMessage] = []
        for raw in raw_entries:
            parsed = PairMessage.from_json(raw)
            if parsed is not None:
                history.append(parsed)
        return history

    async def _increment_pair(self, sender_id: str, target_id: str) -> int:
        key = pair_key(sender_id, target_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, self._settings.pair_window_s)
            return int((await pipe.execute())[0])

    async def _increment(self, key: str, ttl: int) -> int:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, ttl)
            return int((await pipe.execute())[0])

    async def _record_message(
        self, sender_id: str, target_id: str
    ) -> tuple[list[PairMessage], str]:
        """Append the message to the shared pair history and return it.

        Returns the history and the id of the entry that was just written.
        """
        settings = self._settings
        key = pair_messages_key(sender_id, target_id)
        entry_id = uuid4().hex
        entry = PairMessage(
            ts=now_seconds(), sender=sender_id, toxicity=0.0, entry_id=entry_id
        )
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.lpush(key, entry.to_json())
            pipe.ltrim(key, 0, settings.pair_message_limit - 1)
            pipe.expire(key, settings.pair_window_s)
            await pipe.execute()
        return await self._pair_history(sender_id, target_id), entry_id

    async def _register_pileon(self, sender_id: str, target_id: str) -> int:
        """Add this sender to the target's pile-on set; return distinct senders."""
        settings = self._settings
        now = now_seconds()
        key = pileon_key(target_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, "-inf", now - settings.pileon_window_s)
            pipe.zadd(key, {sender_id: float(now)})
            pipe.zcard(key)
            pipe.pexpire(key, settings.pileon_window_s * 2000)
            results = await pipe.execute()
        return int(results[2])
