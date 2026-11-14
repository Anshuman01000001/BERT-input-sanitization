"""Behaviour score with time decay and the escalation ladder.

The score is a rolling measure of how much trouble a player has caused. It
decays over time (one point per hour by default) so that an old, isolated
offence eventually stops counting, and it drives the escalation ladder:

    allow -> warn -> flag for review -> temp mute (10 min, 1 hr, 24 hr)
          -> human decides on ban

Writes use ``WATCH`` for optimistic locking: the decayed value depends on the
value read, so read-modify-write has to be guarded against a concurrent
update from another worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.config import Settings
from app.redis_client import mute_key, now_seconds, score_key

logger = logging.getLogger(__name__)

# How many times to retry a WATCH-protected update before giving up on the
# optimistic lock. Contention on a single player's key is rare.
MAX_WATCH_RETRIES = 8


@dataclass(frozen=True, slots=True)
class Escalation:
    """Where a player sits on the escalation ladder."""

    level: str  # none | warn | flag | mute | ban_review
    mute_seconds: int | None = None


class BehaviorScorer:
    """Maintains per-player behaviour scores in Redis."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings

    async def apply(self, player_id: str, points: int) -> int:
        """Add ``points`` (after applying decay) and return the new total."""
        settings = self._settings
        key = score_key(player_id)
        now = now_seconds()

        for _ in range(MAX_WATCH_RETRIES):
            async with self._redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.hgetall(key)
                    current = int(raw.get("points", 0))
                    updated_at = int(raw.get("updated_at", now))
                    decayed = self._decayed(current, updated_at, now)
                    new_value = max(0, decayed + points)

                    pipe.multi()
                    pipe.hset(key, mapping={"points": new_value, "updated_at": now})
                    pipe.expire(key, settings.score_ttl_s)
                    await pipe.execute()
                    return new_value
                except WatchError:
                    continue

        logger.warning("score update for %s kept losing the optimistic lock", player_id)
        return await self._apply_unlocked(player_id, points, now)

    async def get(self, player_id: str) -> int:
        """Current score including decay, without persisting the decay."""
        now = now_seconds()
        raw = await self._redis.hgetall(score_key(player_id))
        if not raw:
            return 0
        current = int(raw.get("points", 0))
        updated_at = int(raw.get("updated_at", now))
        return max(0, self._decayed(current, updated_at, now))

    async def escalation_for(self, player_id: str, score: int) -> Escalation:
        """Place ``player_id`` on the escalation ladder and manage the mute tier.

        Leaving the mute band clears the marker, so a player who drops back
        down and misbehaves again starts from the shortest mute again.
        """
        settings = self._settings
        raw = await self._redis.hgetall(score_key(player_id))
        mute_tier = int(raw.get("mute_tier", 0))

        if score >= settings.score_ban_review_at:
            await self._clear_mute_tier(player_id, mute_tier)
            return Escalation("ban_review")

        if score >= settings.score_mute_at:
            stored = int(raw.get("mute_duration", 0))
            if mute_tier >= 1 and stored > 0:
                return Escalation("mute", stored)
            counter = int(await self._redis.get(mute_key(player_id)) or 0)
            index = min(counter, len(settings.mute_durations_s) - 1)
            return Escalation("mute", settings.mute_durations_s[index])

        await self._clear_mute_tier(player_id, mute_tier)
        if score >= settings.score_flag_at:
            return Escalation("flag")
        if score >= settings.score_warn_at:
            return Escalation("warn")
        return Escalation("none")

    async def note_mute(self, player_id: str, mute_seconds: int) -> None:
        """Record that the player has just entered the mute band.

        The mute counter only advances when the player *enters* the band, so
        the mute length escalates 10 min -> 1 hr -> 24 hr across separate
        incidents rather than across every message of one incident.
        """
        settings = self._settings
        key = score_key(player_id)
        for _ in range(MAX_WATCH_RETRIES):
            async with self._redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.hgetall(key)
                    if int(raw.get("mute_tier", 0)) >= 1:
                        await pipe.unwatch()
                        return
                    pipe.multi()
                    pipe.hset(
                        key, mapping={"mute_tier": 1, "mute_duration": mute_seconds}
                    )
                    pipe.expire(key, settings.score_ttl_s)
                    pipe.incr(mute_key(player_id))
                    pipe.expire(mute_key(player_id), settings.mute_ttl_s)
                    await pipe.execute()
                    return
                except WatchError:
                    continue
        logger.warning("could not record mute tier for %s", player_id)

    async def _clear_mute_tier(self, player_id: str, mute_tier: int) -> None:
        if mute_tier < 1:
            return
        await self._redis.hset(
            score_key(player_id), mapping={"mute_tier": 0, "mute_duration": 0}
        )

    def _decayed(self, current: int, updated_at: int, now: int) -> int:
        """Apply time decay to a stored score."""
        settings = self._settings
        elapsed = max(0, now - updated_at)
        intervals = elapsed // settings.score_decay_interval_s
        return current - intervals * settings.score_decay_points

    async def _apply_unlocked(self, player_id: str, points: int, now: int) -> int:
        """Last-resort write when the optimistic lock could not be taken."""
        key = score_key(player_id)
        raw = await self._redis.hgetall(key)
        current = int(raw.get("points", 0))
        updated_at = int(raw.get("updated_at", now))
        new_value = max(0, self._decayed(current, updated_at, now) + points)
        await self._redis.hset(key, mapping={"points": new_value, "updated_at": now})
        await self._redis.expire(key, self._settings.score_ttl_s)
        return new_value
