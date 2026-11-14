"""Latency and throughput benchmark for the moderation service.

Model-level (no server needed):

    .venv/bin/python tools/benchmark.py
    .venv/bin/python tools/benchmark.py --messages 1000 --threads 4
    .venv/bin/python tools/benchmark.py --compare-fp32

End-to-end against a running instance:

    .venv/bin/python tools/benchmark.py --url http://localhost:8000 \
        --api-key secret --messages 500

The handoff quotes 10-30 ms per short message as an *unbenchmarked estimate*;
run this on the target EPYC box to replace it with a measured number.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The settings resolve `models/` relative to the working directory.
os.chdir(REPO_ROOT)

from app.config import Settings  # noqa: E402
from app.model import ToxicityModel  # noqa: E402

# Typical game-chat lines: short, mostly benign, some not.
MESSAGES = [
    "gg",
    "nice shot",
    "wp",
    "let's queue ranked",
    "who has heal potions",
    "im getting destroyed over here",
    "turn left at the pillar",
    "thanks for the heal",
    "anyone trading shards",
    "that was a good game everyone",
    "i think i have the right build",
    "gl everyone",
]


def percentile(values: list[float], fraction: float) -> float:
    """The ``fraction`` percentile of ``values`` (which need not be sorted)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(name: str, durations_ms: list[float]) -> None:
    """Print the latency distribution of one measurement."""
    if not durations_ms:
        return
    print(
        f"  {name:<34} "
        f"p50 {percentile(durations_ms, 0.50):6.2f} ms   "
        f"p90 {percentile(durations_ms, 0.90):6.2f} ms   "
        f"p99 {percentile(durations_ms, 0.99):6.2f} ms   "
        f"mean {statistics.fmean(durations_ms):6.2f} ms"
    )


def distinct_messages(count: int) -> list[str]:
    """Enough distinct texts that the score cache never serves a repeat."""
    return [f"{MESSAGES[index % len(MESSAGES)]} #{index}" for index in range(count)]


def build_scorer(path: Path, settings: Settings) -> ToxicityModel:
    """Build a scorer around a specific ONNX file.

    The tokenizer and the metadata always come from the settings' model
    directory, so a float graph elsewhere can be benchmarked against the same
    tokenizer.
    """
    scorer = ToxicityModel(
        model_dir=settings.model_dir,
        onnx_file=str(path),
        tokenizer_file=settings.tokenizer_file,
        labels_file=settings.labels_file,
        threads=settings.onnx_threads,
        max_seq_len=settings.max_seq_len,
        cache_size=settings.score_cache_size,
    )
    scorer.load()
    return scorer


def benchmark_model(scorer: ToxicityModel, messages: list[str]) -> None:
    """Measure cold, warm and cached latency, plus sequential throughput."""
    # Cold: the first call for a given input shape pays for graph setup.
    started = time.perf_counter()
    scorer.score(messages[0])
    cold_ms = (time.perf_counter() - started) * 1000
    print(f"  cold start (first inference)       {cold_ms:6.2f} ms")

    # Warm, uncached: every text is distinct, so the cache never hits.
    durations: list[float] = []
    started = time.perf_counter()
    for text in messages:
        call_started = time.perf_counter()
        scorer.score(text)
        durations.append((time.perf_counter() - call_started) * 1000)
    elapsed = time.perf_counter() - started
    summarize("warm, cache miss", durations)

    # Warm, cached: the same text over and over.
    durations = []
    for _ in range(len(messages)):
        call_started = time.perf_counter()
        scorer.score("gg")
        durations.append((time.perf_counter() - call_started) * 1000)
    summarize("warm, cache hit", durations)

    print(
        f"  sequential throughput               "
        f"{len(messages) / elapsed:6.1f} messages/s"
    )


async def benchmark_http(
    url: str, api_key: str, messages: list[str], concurrency: int
) -> None:
    """Measure end-to-end latency against a running instance."""
    import httpx

    headers = {"X-API-Key": api_key} if api_key else {}
    semaphore = asyncio.Semaphore(concurrency)
    durations: list[float] = []
    base = url.rstrip("/")

    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=30) as client:

        async def one(index: int, text: str) -> None:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    "/moderate",
                    json={
                        "text": text,
                        "player_id": f"bench{index}",
                        "channel": "global",
                    },
                )
                response.raise_for_status()
                durations.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        await asyncio.gather(
            *(one(index, text) for index, text in enumerate(messages))
        )
        elapsed = time.perf_counter() - started

    summarize("end-to-end", durations)
    print(
        f"  throughput ({concurrency} concurrent)      "
        f"{len(messages) / elapsed:6.1f} requests/s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=300)
    parser.add_argument("--url", help="Benchmark a running instance over HTTP.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MOD_API_KEY", ""),
        help="API key for --url; defaults to $MOD_API_KEY.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Concurrent HTTP requests."
    )
    parser.add_argument(
        "--compare-fp32",
        action="store_true",
        help="Also benchmark build/fp32/model.onnx if it exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    messages = distinct_messages(args.messages)

    if args.url:
        print(f"end-to-end benchmark against {args.url}")
        asyncio.run(
            benchmark_http(args.url, args.api_key, messages, args.concurrency)
        )
        return 0

    int8_path = settings.model_dir / settings.onnx_file
    print(f"model benchmark: {int8_path}")
    print(f"  threads={settings.onnx_threads}  messages={args.messages}")
    scorer = build_scorer(int8_path, settings)
    benchmark_model(scorer, messages)

    fp32_path = REPO_ROOT / "build" / "fp32" / "model.onnx"
    if args.compare_fp32 and fp32_path.exists():
        print(f"\nfloat graph for comparison: {fp32_path}")
        fp32 = build_scorer(fp32_path, settings)
        benchmark_model(fp32, messages)
    elif args.compare_fp32:
        print(f"\n{fp32_path} not found; run tools/export_model.py first.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
