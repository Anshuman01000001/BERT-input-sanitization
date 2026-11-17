"""Redis key layout and connection helpers.

Every key carries a TTL. The service keeps no durable state, so losing Redis
costs the *stateful* heuristics (rate limiting, duplicate detection,
harassment counters) but never the model-based toxicity scoring.

Atomicity without Lua: redis-py's ``MULTI``/``EXEC`` pipeline runs its
commands as one uninterrupted unit, so a read-then-write sequence inside a
transaction observes a consistent snapshot. For the score, where the written
value depends on the value read, ``WATCH`` provides the optimistic lock.
This keeps the runtime free of a Lua interpreter (and keeps the suite
runnable against fakeredis).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

from redis.asyncio import Redis

from app.config import Settings

logger = logging.getLogger(__name__)


def rate_key(player_id: str) -> str:
    """Sorted set of send timestamps, used for the sliding-window rate limit."""
    return f"rate:{player_id}"


def duplicate_key(player_id: str, msg_hash: str) -> str:
    """Counter of how many times a player sent this exact normalized text."""
    return f"dup:{player_id}:{msg_hash}"


def recent_key(player_id: str) -> str:
    """List of a player's recently sent normalized texts (near-duplicate check)."""
    return f"recent:{player_id}"


def pair_key(sender_id: str, target_id: str) -> str:
    """Counter of messages sent from one player to another."""
    return f"pair:{sender_id}:{target_id}"


def pair_messages_key(player_a: str, player_b: str) -> str:
    """Recent messages exchanged between two players, in either direction.

    The pair is *unordered* so that a target's reply lands in the same
    history as the messages that prompted it - without that, "the target never
    replied" can never be observed.
    """
    first, second = sorted((player_a, player_b))
    return f"pairmsgs:{first}:{second}"


def pileon_key(target_id: str) -> str:
    """Distinct senders recently targeting one player."""
    return f"pileon:{target_id}"


def score_key(player_id: str) -> str:
    """Behaviour score hash: points, updated_at, mute_tier, mute_duration."""
    return f"score:{player_id}"


def mute_key(player_id: str) -> str:
    """How many times the player has been muted (drives mute duration)."""
    return f"mute:{player_id}"


def reports_made_key(reporter_id: str) -> str:
    """How many reports this player has filed (drives reporter trust)."""
    return f"reports_made:{reporter_id}"


def reports_against_key(target_id: str) -> str:
    """How many reports have been filed against this player."""
    return f"reports_against:{target_id}"


def api_limit_key(api_key: str, client_id: str = "global") -> str:
    """Fixed-window counter for API-level rate limiting.

    Keyed by API key *and* client, so one caller cannot spend everyone's
    budget; the key itself never appears in the Redis keyspace.
    """
    digest = hashlib.sha256(f"{api_key}:{client_id}".encode()).hexdigest()[:16]
    return f"apilimit:{digest}"


def report_points_key(reporter_id: str, target_id: str) -> str:
    """How many points one reporter has already contributed to one target."""
    return f"reportpoints:{reporter_id}:{target_id}"


def now_seconds() -> int:
    """Current UTC time as whole epoch seconds."""
    return int(time.time())


def unique_member(timestamp: float) -> str:
    """Unique member value for sorted sets (timestamps can collide)."""
    return f"{timestamp:.6f}-{uuid.uuid4().hex[:8]}"


def create_client(settings: Settings) -> Redis:
    """Build an async Redis client from settings.

    ``MOD_REDIS_URL=fakeredis://`` runs the stateful checks against an
    in-process fake Redis. It is a development convenience for machines with
    no Redis server: the state does not survive the process and is not shared
    between workers, so it is not a substitute for the real thing.
    """
    if settings.redis_url.startswith("fakeredis://"):
        try:
            import fakeredis.aioredis as fakeredis
        except ImportError as exc:
            raise RuntimeError(
                "MOD_REDIS_URL points at fakeredis, but fakeredis is not "
                "installed. Install the development dependencies: "
                "uv pip install -r requirements-dev.txt"
            ) from exc
        logger.info(
            "using an in-process fake redis; the state is not shared or persisted"
        )
        return fakeredis.FakeRedis(decode_responses=True)

    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        # A bounded pool: an unbounded one lets a slow Redis exhaust memory.
        max_connections=32,
        health_check_interval=30,
    )
