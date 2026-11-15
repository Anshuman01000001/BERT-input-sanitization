"""Spam heuristics: rate limiting, duplicate detection, floods and links.

These checks are cheap (string work plus a couple of Redis round-trips), which
is exactly why the pipeline runs them before the classifier: most abuse is
stopped without spending a single CPU cycle on the model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings
from app.normalize import NormalizedMessage, similarity
from app.redis_client import (
    duplicate_key,
    now_seconds,
    rate_key,
    recent_key,
    unique_member,
)

# --- link / invite patterns ---------------------------------------------------
_URL_RE = re.compile(
    r"(?:https?://|ftp://)\S+"
    r"|\bwww\.[^\s]+"
    r"|\b[\w-]+\.(?:com|net|org|io|gg|xyz|ru|top|link|me|co|us|info)\b\S*",
    re.IGNORECASE,
)
_DISCORD_INVITE_RE = re.compile(
    r"discord(?:app)?\.(?:gg|com/invite)/[\w-]+", re.IGNORECASE
)
_EXTERNAL_INVITE_RE = re.compile(
    r"(?:steam://|skype:|t\.me/|telegram\.me/|whatsapp:|tinyurl\.com)",
    re.IGNORECASE,
)

# Signals that send the message to human review rather than muting it.
_REVIEW_ONLY_REASONS = frozenset({"link", "discord_invite", "external_invite"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpamVerdict:
    """Outcome of the spam checks.

    ``action`` is one of ``allow``, ``flag`` or ``shadow_mute``. Spam content
    is shadow-muted (the sender sees their own message, nobody else does)
    rather than blocked outright.
    """

    action: str = "allow"
    reasons: tuple[str, ...] = ()
    detail: dict[str, float] = field(default_factory=dict)
    # False when the stateful checks (rate limiting and duplicates) could
    # not run. The stateless ones - caps floods, character floods, links and
    # invites - always run: they need no Redis, and losing them because
    # Redis is down would be a needless blind spot.
    state_available: bool = True

    @property
    def is_spam(self) -> bool:
        return self.action != "allow"


def caps_ratio(text: str) -> float:
    """Share of alphabetic characters that are uppercase."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def max_char_run(text: str) -> int:
    """Length of the longest run of identical characters."""
    longest = 0
    run = 0
    previous = ""
    for ch in text:
        run = run + 1 if ch == previous else 1
        if run > longest:
            longest = run
        previous = ch
    return longest


def symbol_ratio(text: str) -> float:
    """Share of characters that are neither alphanumeric nor whitespace."""
    if not text:
        return 0.0
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbols / len(text)


def extract_hosts(text: str) -> set[str]:
    """Hostnames of every URL-like token in ``text`` (lowercased, no ``www.``)."""
    hosts: set[str] = set()
    for match in _URL_RE.finditer(text):
        raw = match.group(0)
        candidate = raw if "://" in raw else f"http://{raw}"
        try:
            host = urlparse(candidate).hostname
        except ValueError:
            host = None
        if host:
            hosts.add(host.lower().removeprefix("www."))
    return hosts


class SpamChecker:
    """Rate limiting, duplicate detection and content-pattern spam checks."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings

    async def check(
        self,
        player_id: str,
        message: NormalizedMessage,
        account_age_s: int | None = None,
    ) -> SpamVerdict:
        """Run every spam check and return the combined verdict.

        The stateless checks run first and always; the stateful ones degrade
        to "not available" if Redis is unreachable rather than taking the
        whole verdict down with them.
        """
        pattern_reasons, pattern_detail = self._pattern_signals(message)
        reasons: list[str] = list(pattern_reasons)
        detail: dict[str, float] = dict(pattern_detail)
        action = "allow"
        if _REVIEW_ONLY_REASONS.intersection(pattern_reasons):
            # A single link or invite is worth reviewing, not worth
            # shadow-muting on its own.
            action = "flag"
        elif pattern_reasons:
            action = "shadow_mute"

        state_available = True
        try:
            rate_count = await self._rate_limit_count(player_id, account_age_s)
            if rate_count is not None:
                reasons.append("rate_limit")
                detail["rate_limit_count"] = float(rate_count)
                action = "shadow_mute"

            duplicate, near_duplicate = await self._duplicate_signals(
                player_id, message
            )
            if duplicate is not None:
                reasons.append("duplicate")
                detail["duplicate_count"] = float(duplicate)
                action = "shadow_mute"
            elif near_duplicate is not None:
                reasons.append("near_duplicate")
                detail["near_duplicate_similarity"] = near_duplicate
                action = "shadow_mute"
        except RedisError:
            logger.warning(
                "redis unavailable; the stateful spam checks are disabled",
                exc_info=True,
            )
            state_available = False

        return SpamVerdict(
            action=action,
            reasons=tuple(reasons),
            detail=detail,
            state_available=state_available,
        )

    async def _rate_limit_count(
        self, player_id: str, account_age_s: int | None
    ) -> int | None:
        """Messages already sent in the window, or ``None`` if under the limit.

        New accounts get a tighter limit: the median abuse account is fresh.
        """
        settings = self._settings
        limit = settings.rate_limit_count
        if account_age_s is not None and account_age_s < settings.new_account_age_s:
            limit = settings.new_account_rate_limit

        now = now_seconds()
        window = settings.rate_limit_window_s
        key = rate_key(player_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, "-inf", now - window)
            pipe.zcard(key)
            pipe.zadd(key, {unique_member(now): float(now)})
            pipe.pexpire(key, window * 2000)
            results = await pipe.execute()
        count = int(results[1])
        return count if count >= limit else None

    async def _duplicate_signals(
        self, player_id: str, message: NormalizedMessage
    ) -> tuple[int | None, float | None]:
        """Exact-duplicate count and best near-duplicate similarity."""
        settings = self._settings
        key = duplicate_key(player_id, message.message_hash)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, settings.duplicate_window_s)
            exact_count = int((await pipe.execute())[0])

        best_similarity: float | None = None
        if exact_count == 1:
            recent = await self._redis.lrange(
                recent_key(player_id), 0, settings.recent_messages_kept - 1
            )
            for previous in recent:
                score = similarity(previous, message.normalized)
                if score >= settings.near_duplicate_threshold:
                    best_similarity = score
                    break

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.lpush(recent_key(player_id), message.normalized)
            pipe.ltrim(recent_key(player_id), 0, settings.recent_messages_kept - 1)
            pipe.expire(recent_key(player_id), settings.recent_messages_window_s)
            await pipe.execute()

        duplicate = exact_count if exact_count > 1 else None
        return duplicate, best_similarity

    def _pattern_signals(
        self, message: NormalizedMessage
    ) -> tuple[list[str], dict[str, float]]:
        """Stateless content checks: caps, floods, length and links."""
        settings = self._settings
        text = message.folded
        reasons: list[str] = []
        detail: dict[str, float] = {}

        caps = caps_ratio(text)
        if len(text) >= settings.min_caps_length and caps > settings.max_caps_ratio:
            reasons.append("caps_flood")
            detail["caps_ratio"] = caps

        run = max_char_run(text)
        if run > settings.max_single_char_run:
            reasons.append("char_flood")
            detail["max_char_run"] = float(run)

        symbols = symbol_ratio(text)
        if symbols > settings.max_symbol_ratio:
            reasons.append("symbol_flood")
            detail["symbol_ratio"] = symbols

        if len(message.normalized) > settings.max_spam_message_chars:
            reasons.append("length_flood")
            detail["message_length"] = float(len(message.normalized))

        if _DISCORD_INVITE_RE.search(text):
            reasons.append("discord_invite")
        elif _EXTERNAL_INVITE_RE.search(text):
            reasons.append("external_invite")
        else:
            hosts = extract_hosts(text)
            if hosts:
                blocked = hosts - set(settings.allowlisted_link_hosts)
                if blocked:
                    reasons.append("link")
                    detail["link_count"] = float(len(blocked))

        return reasons, detail
