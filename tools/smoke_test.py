"""Smoke test a running moderation instance.

Unlike the test suite (which exercises the code in-process) this drives a
deployed service over HTTP, so it is what you want on the VPS before you
point the game server at it:

    .venv/bin/python tools/smoke_test.py --url http://localhost:8000 \
        --api-key "$MOD_API_KEY"

Exits non-zero if any check fails, so it can be used as a deployment gate.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

PASSED = "\033[32mPASS\033[0m"
FAILED = "\033[31mFAIL\033[0m"
SKIPPED = "\033[33mSKIP\033[0m"


class SmokeFailure(Exception):
    """A check did not hold."""


async def moderate(client, url: str, api_key: str, **payload) -> dict:
    """POST one message and return the decoded decision."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    response = await client.post(
        f"{url.rstrip('/')}/moderate", json=payload, headers=headers
    )
    if response.status_code != 200:
        raise SmokeFailure(f"HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


async def run_checks(client, url: str, api_key: str) -> list[tuple[str, str, str]]:
    """Run every check; return (name, status, detail) for each.

    ``status`` is ``pass``, ``fail`` or ``skip``. A check that needs Redis is
    skipped, not failed, when Redis is unreachable: the service is supposed
    to keep classifying without it.
    """
    results: list[tuple[str, str, str]] = []

    def record(name: str, status: str, detail: str = "") -> None:
        results.append((name, status, detail))

    # --- health -----------------------------------------------------------
    redis_ok = False
    try:
        response = await client.get(f"{url.rstrip('/')}/health")
        body = response.json()
        redis_ok = bool(body.get("redis_ok"))
        healthy = response.status_code == 200 and body.get("model_loaded") is True
        record(
            "health: model is loaded",
            "pass" if healthy else "fail",
            f"redis_ok={body.get('redis_ok')} status={body.get('status')}",
        )
    except Exception as exc:  # noqa: BLE001 - report, do not crash the run
        record("health: model is loaded", "fail", str(exc))

    # --- authentication ---------------------------------------------------
    try:
        response = await client.post(
            f"{url.rstrip('/')}/moderate",
            json={"text": "hello", "player_id": "smoke"},
            headers={"Content-Type": "application/json", "X-API-Key": "definitely-wrong"},
        )
        record(
            "auth: a wrong key is rejected",
            "pass" if response.status_code == 401 else "fail",
            f"status={response.status_code}",
        )
    except Exception as exc:  # noqa: BLE001
        record("auth: a wrong key is rejected", "fail", str(exc))

    # --- decisions --------------------------------------------------------
    cases = [
        ("clean chat is allowed", "gg, nice shot!", "allow"),
        ("a threat is blocked", "I will find you and break your legs", "block"),
        ("an insult is blocked", "you are a disgusting idiot", "block"),
        ("a mild trait is held back", "go away nobody likes you", "shadow_mute"),
        ("a discord invite is flagged", "join my discord discord.gg/abcd1234", "flag"),
    ]
    for index, (name, text, expected) in enumerate(cases):
        try:
            body = await moderate(
                client, url, api_key, text=text, player_id=f"smoke{index}"
            )
            actual = body.get("action")
            record(
                name,
                "pass" if actual == expected else "fail",
                f"action={actual} reasons={body.get('reasons')}",
            )
        except Exception as exc:  # noqa: BLE001
            record(name, "fail", str(exc))

    # --- a threat that also carries a link must still be blocked ----------
    try:
        body = await moderate(
            client,
            url,
            api_key,
            text="I will find you and break your legs http://sketchy-site.org/x",
            player_id="smokelink",
        )
        record(
            "a threat with a link is still blocked",
            "pass" if body.get("action") == "block" else "fail",
            f"action={body.get('action')}",
        )
    except Exception as exc:  # noqa: BLE001
        record("a threat with a link is still blocked", "fail", str(exc))

    # --- the behaviour score accumulates ----------------------------------
    if not redis_ok:
        record("the behaviour score grows", "skip", "redis is unreachable")
    else:
        try:
            player = "smokescore"
            first = await moderate(
                client, url, api_key, text="you are a disgusting idiot", player_id=player
            )
            second = await moderate(
                client, url, api_key, text="you are a disgusting idiot", player_id=player
            )
            grew = second["player_score"] > first["player_score"]
            record(
                "the behaviour score grows",
                "pass" if grew else "fail",
                f"{first['player_score']} -> {second['player_score']}",
            )
        except Exception as exc:  # noqa: BLE001
            record("the behaviour score grows", "fail", str(exc))

    # --- a report about a stranger must not move the target ---------------
    if not redis_ok:
        record("a report about a stranger moves nothing", "skip", "redis is unreachable")
    else:
        try:
            body = await moderate(
                client,
                url,
                api_key,
                text="reporting this behaviour",
                player_id="smokereporter",
                target_id="smokestranger",
                is_report=True,
            )
            record(
                "a report about a stranger moves nothing",
                "pass" if body.get("target_score") in (0, None) else "fail",
                f"target_score={body.get('target_score')} reasons={body.get('reasons')}",
            )
        except Exception as exc:  # noqa: BLE001
            record("a report about a stranger moves nothing", "fail", str(exc))

    # --- an oversized body is refused -------------------------------------
    try:
        response = await client.post(
            f"{url.rstrip('/')}/moderate",
            content=b"x" * 20_000,
            headers={"Content-Type": "application/json"},
        )
        record(
            "an oversized body is refused",
            "pass" if response.status_code in (401, 413) else "fail",
            f"status={response.status_code}",
        )
    except Exception as exc:  # noqa: BLE001
        record("an oversized body is refused", "fail", str(exc))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MOD_API_KEY", ""),
        help="API key; defaults to $MOD_API_KEY.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    import httpx

    print(f"Smoke testing {args.url}\n")
    async with httpx.AsyncClient(timeout=30) as client:
        results = await run_checks(client, args.url, args.api_key)

    failures = 0
    skipped = 0
    for name, status, detail in results:
        marker = {"pass": PASSED, "fail": FAILED, "skip": SKIPPED}[status]
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{marker}] {name}{suffix}")
        if status == "fail":
            failures += 1
        elif status == "skip":
            skipped += 1

    total = len(results)
    print(f"\n{total - failures - skipped}/{total} checks passed"
          + (f", {skipped} skipped" if skipped else ""))
    if failures:
        print(
            "The instance is not behaving as expected; "
            "do not point the game server at it."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
