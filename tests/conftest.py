"""Shared pytest fixtures.

The suite runs against ``fakeredis`` instead of a real Redis server. The
service uses only commands fakeredis implements (strings, hashes, sorted sets,
lists, pipelines and WATCH), so the behaviour under test matches production.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fakeredis.aioredis as fakeredis
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    """Settings tuned for tests: predictable key and a real model directory."""
    return Settings(
        # Long enough to satisfy the startup key policy.
        api_key="test-key-for-the-suite",
        require_api_key=True,
        model_dir=REPO_ROOT / "models",
        enable_demo=True,
    )


@pytest.fixture
async def redis_client():
    """An in-memory Redis with the same wire behaviour as the real one."""
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()
