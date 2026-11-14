"""Tests for the spam heuristics."""

from __future__ import annotations

import pytest

from app.normalize import normalize
from app.spam import SpamChecker, caps_ratio, extract_hosts, max_char_run, symbol_ratio

# Distinct enough that the near-duplicate detector does not fire on them.
DISTINCT = [
    "alpha bravo",
    "charlie delta",
    "echo foxtrot",
    "golf hotel",
    "india juliet",
    "kilo lima",
]


@pytest.fixture
def checker(settings, redis_client) -> SpamChecker:
    return SpamChecker(redis_client, settings)


class TestHelpers:
    def test_caps_ratio(self):
        assert caps_ratio("GGGG") == 1.0
        assert caps_ratio("gggg") == 0.0
        assert caps_ratio("") == 0.0

    def test_max_char_run(self):
        assert max_char_run("aaabbb") == 3
        assert max_char_run("abc") == 1
        assert max_char_run("") == 0

    def test_symbol_ratio(self):
        assert symbol_ratio("😀😀😀😀") == 1.0
        assert symbol_ratio("hello world") == 0.0

    def test_extract_hosts_strips_www(self):
        assert extract_hosts("visit https://www.example.com/x now") == {"example.com"}

    def test_extract_hosts_finds_bare_domains(self):
        assert extract_hosts("go to sketchy-site.org/guide") == {"sketchy-site.org"}


class TestRateLimit:
    async def test_allows_messages_under_the_limit(self, checker):
        for text in DISTINCT[:5]:
            verdict = await checker.check("p1", normalize(text))
            assert "rate_limit" not in verdict.reasons

    async def test_blocks_the_sixth_message_in_the_window(self, checker):
        for text in DISTINCT[:5]:
            await checker.check("p1", normalize(text))
        verdict = await checker.check("p1", normalize(DISTINCT[5]))
        assert "rate_limit" in verdict.reasons
        assert verdict.action == "shadow_mute"

    async def test_new_accounts_get_a_tighter_limit(self, checker):
        for text in DISTINCT[:3]:
            await checker.check("p2", normalize(text), account_age_s=60)
        verdict = await checker.check("p2", normalize(DISTINCT[3]), account_age_s=60)
        assert "rate_limit" in verdict.reasons

    async def test_established_accounts_keep_the_normal_limit(self, checker):
        for text in DISTINCT[:3]:
            await checker.check("p3", normalize(text), account_age_s=86400)
        verdict = await checker.check("p3", normalize(DISTINCT[3]), account_age_s=86400)
        assert "rate_limit" not in verdict.reasons

    async def test_players_are_limited_independently(self, checker):
        for text in DISTINCT:
            await checker.check("p4", normalize(text))
        verdict = await checker.check("p5", normalize("hello"))
        assert "rate_limit" not in verdict.reasons


class TestDuplicates:
    async def test_first_identical_message_is_not_a_duplicate(self, checker):
        verdict = await checker.check("p6", normalize("same message here"))
        assert "duplicate" not in verdict.reasons

    async def test_second_identical_message_is_a_duplicate(self, checker):
        await checker.check("p6", normalize("same message here"))
        verdict = await checker.check("p6", normalize("same message here"))
        assert "duplicate" in verdict.reasons
        assert verdict.action == "shadow_mute"

    async def test_near_duplicate_is_detected(self, checker):
        await checker.check("p7", normalize("join my discord now"))
        verdict = await checker.check("p7", normalize("join my discord now!"))
        assert "near_duplicate" in verdict.reasons

    async def test_different_players_do_not_share_duplicate_counters(self, checker):
        await checker.check("p8", normalize("same message here"))
        verdict = await checker.check("p9", normalize("same message here"))
        assert "duplicate" not in verdict.reasons


class TestContentPatterns:
    async def test_caps_flood_is_flagged(self, checker):
        verdict = await checker.check("p10", normalize("GGGGGGGGGGGGGGGG"))
        assert "caps_flood" in verdict.reasons

    async def test_normal_caps_amount_is_not_flagged(self, checker):
        verdict = await checker.check("p11", normalize("GG that was a good round"))
        assert "caps_flood" not in verdict.reasons

    async def test_character_flood_is_flagged(self, checker):
        verdict = await checker.check("p12", normalize("a" * 40))
        assert "char_flood" in verdict.reasons

    async def test_emoji_flood_is_flagged(self, checker):
        verdict = await checker.check("p13", normalize("😀" * 20))
        assert "symbol_flood" in verdict.reasons

    async def test_discord_invite_is_detected(self, checker):
        verdict = await checker.check("p14", normalize("join discord.gg/abcd1234"))
        assert "discord_invite" in verdict.reasons
        assert verdict.action == "flag"

    async def test_external_invite_is_detected(self, checker):
        verdict = await checker.check("p15", normalize("add me on steam://friends/1"))
        assert "external_invite" in verdict.reasons
        assert verdict.action == "flag"

    async def test_allowlisted_host_is_not_a_link(self, checker):
        verdict = await checker.check("p16", normalize("look at example.com/guide"))
        assert "link" not in verdict.reasons

    async def test_other_host_is_a_link(self, checker):
        verdict = await checker.check("p17", normalize("look at sketchy-site.org/guide"))
        assert "link" in verdict.reasons

    async def test_a_plain_link_only_is_reviewed_not_muted(self, checker):
        verdict = await checker.check("p18", normalize("look at sketchy-site.org/guide"))
        assert verdict.action == "flag"
