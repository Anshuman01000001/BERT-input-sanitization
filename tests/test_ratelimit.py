"""Tests for the API rate limiter."""

from __future__ import annotations

import asyncio

from redis.exceptions import ConnectionError as RedisConnectionError

from app.ratelimit import ApiRateLimiter, LocalRateLimiter


class BrokenRedis:
    """A Redis client whose every command fails."""

    def pipeline(self, *args, **kwargs):
        raise RedisConnectionError("redis is down")


class TestLocalRateLimiter:
    def test_allows_up_to_the_limit(self):
        limiter = LocalRateLimiter()
        assert limiter.allow("client", limit=3, window_s=60) is True
        assert limiter.allow("client", limit=3, window_s=60) is True
        assert limiter.allow("client", limit=3, window_s=60) is True

    def test_blocks_beyond_the_limit(self):
        limiter = LocalRateLimiter()
        for _ in range(3):
            limiter.allow("client", limit=3, window_s=60)
        assert limiter.allow("client", limit=3, window_s=60) is False

    def test_clients_are_counted_separately(self):
        limiter = LocalRateLimiter()
        for _ in range(3):
            limiter.allow("noisy", limit=3, window_s=60)
        assert limiter.allow("quiet", limit=3, window_s=60) is True

    def test_hits_outside_the_window_are_forgotten(self):
        limiter = LocalRateLimiter()
        for _ in range(3):
            limiter.allow("client", limit=3, window_s=0)
        # A zero-length window means every earlier hit has expired.
        assert limiter.allow("client", limit=3, window_s=0) is True


class TestApiRateLimiter:
    async def test_allows_traffic_under_the_limit(self, settings, redis_client):
        limiter = ApiRateLimiter(redis_client, settings)
        for _ in range(settings.api_rate_limit_count):
            assert await limiter.allow("client") is True

    async def test_blocks_a_client_that_exceeds_its_limit(self, settings, redis_client):
        limiter = ApiRateLimiter(redis_client, settings)
        for _ in range(settings.api_rate_limit_count):
            await limiter.allow("noisy")
        assert await limiter.allow("noisy") is False

    async def test_one_client_cannot_spend_another_clients_budget(
        self, settings, redis_client
    ):
        limiter = ApiRateLimiter(redis_client, settings)
        for _ in range(settings.api_rate_limit_count):
            await limiter.allow("noisy")
        assert await limiter.allow("quiet") is True

    async def test_the_global_ceiling_applies(self, settings, redis_client):
        settings = settings.model_copy(
            update={"api_rate_limit_count": 10_000, "api_rate_limit_global_count": 3}
        )
        limiter = ApiRateLimiter(redis_client, settings)
        assert await limiter.allow("a") is True
        assert await limiter.allow("b") is True
        assert await limiter.allow("c") is True
        assert await limiter.allow("d") is False

    async def test_it_falls_back_to_a_local_limiter_when_redis_is_down(self, settings):
        limiter = ApiRateLimiter(BrokenRedis(), settings)
        for _ in range(settings.api_rate_limit_count):
            assert await limiter.allow("client") is True
        assert await limiter.allow("client") is False

    async def test_the_fallback_still_separates_clients(self, settings):
        limiter = ApiRateLimiter(BrokenRedis(), settings)
        for _ in range(settings.api_rate_limit_count):
            await limiter.allow("noisy")
        assert await limiter.allow("quiet") is True


class TestWindowRollover:
    """The window has to expire, not slide forward on every request."""

    async def test_steady_traffic_is_not_locked_out(self, settings, redis_client):
        settings = settings.model_copy(update={"api_rate_limit_count": 5})
        limiter = ApiRateLimiter(redis_client, settings)
        for _ in range(2):
            for _ in range(5):
                assert await limiter.allow("steady") is True
            # Let the window close before the next burst.
            await asyncio.sleep(settings.api_rate_limit_window_s + 0.1)

    async def test_steady_traffic_is_not_locked_out_without_redis(self, settings):
        settings = settings.model_copy(update={"api_rate_limit_count": 5})
        limiter = ApiRateLimiter(BrokenRedis(), settings)
        for _ in range(2):
            for _ in range(5):
                assert await limiter.allow("steady") is True
            await asyncio.sleep(settings.api_rate_limit_window_s + 0.1)
