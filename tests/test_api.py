"""End-to-end tests for the HTTP API.

These run the real quantized ONNX model, so they cover the export as well as
the service.
"""

from __future__ import annotations

import fakeredis.aioredis as fakeredis
import httpx
import pytest

import app.main as app_main

# Distinct enough that the near-duplicate detector does not fire on them.
BURST = [
    "join the ranked queue tonight",
    "turn left at the broken pillar",
    "who still has the boons",
    "nice heal on me thanks",
    "reloading the sniper now",
    "anyone trading rare shards",
]

HARASSMENT = [
    "why do you keep ignoring me",
    "answer me right now please",
    "i can see you online",
    "you really are rude today",
]


@pytest.fixture
async def client(settings, monkeypatch):
    """An HTTP client wired to the app, with an in-memory Redis."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "create_client", lambda _settings: fake)
    async with app_main.app.router.lifespan_context(app_main.app):
        transport = httpx.ASGITransport(app=app_main.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-API-Key": "test-key-for-the-suite"},
        ) as http:
            yield http
    await fake.aclose()


async def post(client, **payload) -> dict:
    response = await client.post("/moderate", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


class TestHealth:
    async def test_health_reports_the_model_and_redis_are_ready(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["redis_ok"] is True
        assert body["model_id"] == "unitary/toxic-bert"


class TestAuth:
    async def test_a_missing_key_is_rejected(self, client):
        response = await client.post(
            "/moderate",
            json={"text": "hello", "player_id": "p1"},
            headers={"X-API-Key": ""},
        )
        assert response.status_code == 401

    async def test_the_wrong_key_is_rejected(self, client):
        response = await client.post(
            "/moderate",
            json={"text": "hello", "player_id": "p1"},
            headers={"X-API-Key": "wrong"},
        )
        assert response.status_code == 401


class TestModeration:
    async def test_a_clean_message_is_allowed(self, client):
        body = await post(client, text="gg, nice shot!", player_id="clean")
        assert body["action"] == "allow"
        assert body["scores"]["toxic"] < 0.5
        assert body["player_score"] == 0

    async def test_a_threat_is_blocked(self, client):
        body = await post(
            client, text="I will find you and break your legs", player_id="threat"
        )
        assert body["action"] == "block"
        assert any(reason.startswith("threat:") for reason in body["reasons"])
        assert body["player_score"] > 0

    async def test_a_spam_burst_is_shadow_muted(self, client):
        for text in BURST:
            body = await post(client, text=text, player_id="spammer")
        assert body["action"] == "shadow_mute"
        assert "rate_limit" in body["reasons"]

    async def test_a_repeated_message_is_shadow_muted(self, client):
        await post(client, text="buy cheap skins now", player_id="repeater")
        body = await post(client, text="buy cheap skins now", player_id="repeater")
        assert body["action"] == "shadow_mute"
        assert "duplicate" in body["reasons"]

    async def test_unanswered_targeting_is_shadow_muted(self, client):
        for text in HARASSMENT:
            body = await post(
                client, text=text, player_id="harasser", target_id="victim"
            )
        assert body["action"] == "shadow_mute"
        assert "unanswered_targeting" in body["reasons"]

    async def test_a_threat_is_still_blocked_when_it_also_carries_a_link(self, client):
        """A cheap "flag" verdict must not stop the classifier running."""
        body = await post(
            client,
            text="I will find you and break your legs http://sketchy-site.org/x",
            player_id="mixed",
        )
        assert body["action"] == "block"
        assert any(reason.startswith("threat:") for reason in body["reasons"])

    async def test_a_threat_is_still_blocked_during_a_spam_burst(self, client):
        body = await post(
            client,
            text="I will find you and break your legs",
            player_id="bursty",
        )
        assert body["action"] == "block"

    async def test_the_player_score_grows_with_repeat_offences(self, client):
        first = await post(
            client, text="you are a disgusting idiot", player_id="repeat"
        )
        second = await post(
            client, text="you are a disgusting idiot", player_id="repeat"
        )
        assert second["player_score"] > first["player_score"]

    async def test_a_confirmed_report_raises_the_target_score(self, client):
        # The culprit's message to the reporter is the exchange the report is
        # about, and its toxicity is what corroborates the report.
        await post(
            client,
            text="you are a disgusting idiot",
            player_id="culprit",
            target_id="reporter",
        )
        body = await post(
            client,
            text="reporting this behaviour",
            player_id="reporter",
            target_id="culprit",
            is_report=True,
        )
        assert body["target_score"] == 12
        assert "confirmed_report" in body["reasons"]

    async def test_a_report_about_a_stranger_counts_for_nothing(self, client):
        await post(client, text="you are a disgusting idiot", player_id="quiet")
        body = await post(
            client,
            text="reporting this behaviour",
            player_id="reporter2",
            target_id="quiet",
            is_report=True,
        )
        assert body["target_score"] == 2
        assert "report_no_history" in body["reasons"]

    async def test_a_report_without_corroboration_moves_nothing(self, client):
        # The target has spoken to the reporter, but harmlessly.
        await post(
            client,
            text="good game, well played",
            player_id="polite",
            target_id="reporter4",
        )
        body = await post(
            client,
            text="reporting this behaviour",
            player_id="reporter4",
            target_id="polite",
            is_report=True,
        )
        assert body["target_score"] == 0
        assert "report" in body["reasons"]


class TestValidation:
    async def test_an_over_long_message_is_rejected(self, client):
        response = await client.post(
            "/moderate", json={"text": "x" * 500, "player_id": "p1"}
        )
        assert response.status_code == 422

    async def test_an_unknown_field_is_rejected(self, client):
        response = await client.post(
            "/moderate", json={"text": "hi", "player_id": "p1", "surprise": 1}
        )
        assert response.status_code == 422

    async def test_a_player_id_that_could_escape_the_redis_namespace_is_rejected(self, client):
        response = await client.post(
            "/moderate", json={"text": "hi", "player_id": "p:evil"}
        )
        assert response.status_code == 422

    async def test_a_channel_with_control_characters_is_rejected(self, client):
        response = await client.post(
            "/moderate", json={"text": "hi", "player_id": "p1", "channel": "a\nb"}
        )
        assert response.status_code == 422

    async def test_a_lone_surrogate_is_rejected_not_crashed(self, client):
        # JSON permits "\ud800"; UTF-8 encoding of it raises.
        response = await client.post(
            "/moderate", content=b'{"text": "\\ud800", "player_id": "p1"}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    async def test_an_oversized_body_is_rejected(self, client):
        response = await client.post(
            "/moderate",
            content=b"x" * 5000,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413


class TestDemoConfig:
    async def test_config_never_exposes_the_api_key(self, client):
        response = await client.get("/demo/config")
        assert response.status_code == 200
        body = response.json()
        assert "api_key" not in body
        assert "require_api_key" not in body
        assert body["model_id"] == "unitary/toxic-bert"

    async def test_the_demo_is_hidden_when_it_is_switched_off(
        self, client, settings, monkeypatch
    ):
        monkeypatch.setattr(
            app_main.app.state,
            "settings",
            settings.model_copy(update={"enable_demo": False}),
        )
        assert (await client.get("/demo")).status_code == 404
        assert (await client.get("/demo/config")).status_code == 404


class TestSecurityMiddleware:
    async def test_an_oversized_body_is_rejected_before_authentication(self, client):
        response = await client.post(
            "/moderate",
            content=b"x" * 5000,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413

    async def test_a_duplicate_api_key_header_is_rejected(self, client):
        response = await client.post(
            "/moderate",
            json={"text": "hi", "player_id": "p1"},
            headers=[("x-api-key", "a"), ("x-api-key", "b")],
        )
        assert response.status_code == 400

    async def test_a_chunked_body_over_the_limit_is_rejected(self, client):
        async def chunks():
            yield b"x" * 6000

        response = await client.post(
            "/moderate",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
