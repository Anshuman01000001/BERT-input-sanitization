"""API-level rate limiting.

Redis holds the authoritative counters: one per calling client plus a global
ceiling. Without the per-client dimension a single misbehaving caller
consumes the whole budget, because every caller shares one API key.

When Redis is unreachable the service falls back to a local, in-process
counter rather than skipping the limit. The classifier is the expensive
resource here, so "no limit" is exactly the wrong failure mode: it turns a
degraded Redis into unbounded CPU burn.
"""

from __future__ import annotations

import logging
import threading
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings
from app.logging_config import RECURRING_ERRORS
from app.redis_client import api_limit_key

logger = logging.getLogger(__name__)

# Above this many tracked clients the local limiter prunes expired entries.
LOCAL_LIMITER_MAX_KEYS = 10_000


class LocalRateLimiter:
    """A fixed-window counter kept in the process, used when Redis is down."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_s: int) -> bool:
        """Record a hit and report whether it is within the limit."""
        now = time.monotonic()
        cutoff = now - window_s
        with self._lock:
            hits = [hit for hit in self._hits.get(key, ()) if hit > cutoff]
            if len(hits) >= limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            if len(self._hits) > LOCAL_LIMITER_MAX_KEYS:
                self._hits = {
                    name: times
                    for name, times in self._hits.items()
                    if times and times[-1] > cutoff
                }
            return True


class ApiRateLimiter:
    """Per-client API rate limiting with a global ceiling."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings
        self._local = LocalRateLimiter()

    async def allow(self, client_id: str) -> bool:
        """Record a request and report whether it is within the limit."""
        settings = self._settings
        window = settings.api_rate_limit_window_s
        client_key = api_limit_key(settings.api_key, client_id)
        global_key = api_limit_key(settings.api_key, "global")

        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                # SET NX EX sets the TTL once, when the counter is created.
                # Re-issuing EXPIRE on every request would push the window
                # forward forever and 429 a client that never goes quiet.
                # The initial value is 0 so that the first INCR leaves the
                # counter at 1 rather than 2.
                pipe.set(client_key, 0, nx=True, ex=window)
                pipe.incr(client_key)
                pipe.set(global_key, 0, nx=True, ex=window)
                pipe.incr(global_key)
                results = await pipe.execute()
        except RedisError:
            if RECURRING_ERRORS.should_report("ratelimit"):
                logger.warning(
                    "redis unavailable; rate limiting locally instead", exc_info=True
                )
            if not self._local.allow(
                client_key, settings.api_rate_limit_count, window
            ):
                return False
            return self._local.allow(
                global_key, settings.api_rate_limit_global_count, window
            )

        client_count = int(results[1])
        global_count = int(results[3])
        return (
            client_count <= settings.api_rate_limit_count
            and global_count <= settings.api_rate_limit_global_count
        )
