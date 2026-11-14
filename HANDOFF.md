# Handoff: Game Chat Moderation API (Demo)

Paste this whole document into the next LLM as context, then continue from "Next steps".

## 1. Goal

Build a **demo** of a moderation service for in-game text chat. It must handle:
- **Toxic content** (insults, threats, slurs, obscenity, identity hate)
- **Spam** (flooding, duplicates, links/invites)
- **Harassment** (repeated targeting of one player, pile-ons, escalation)

The demo will be presented, so it should be explainable (clear pipeline, logged reasons) and easy to show live.

## 2. Constraints and environment

- Deployment: Docker container on a VPS with an **AMD EPYC CPU, no GPU**
- Interface: **REST API** (game server calls it per chat message)
- Must be lightweight and low-latency (short chat messages)
- Dev machine: Arch Linux desktop (Docker for local testing)
- Language/stack: Python, FastAPI, ONNX Runtime, Redis

## 3. Decisions already made

| Area | Decision |
|---|---|
| Content model | `unitary/toxic-bert` (BERT-base, ~110M params, the model behind Detoxify), multi-label: toxic, severe_toxic, obscene, threat, insult, identity_hate |
| Lighter fallback | `martin-ha/toxic-comment-model` (DistilBERT, ~66M), faster but less accurate |
| Inference | Export to ONNX, int8 dynamic quantization (Hugging Face Optimum or `onnxruntime.quantization.quantize_dynamic`), serve with ONNX Runtime |
| API | FastAPI, model and tokenizer loaded once at startup |
| State | Redis with TTLs on all keys |
| Spam response | Shadow mute (sender sees their own messages, nobody else does) |
| Scoring | Per-player behavior score with time decay, escalating actions |

## 4. Pipeline

```
Request -> 1. Sanitize -> 2. Spam rules -> 3. Harassment tracker -> 4. BERT -> 5. Decision + player score
```

Cheap checks run first so most abuse is caught before the model uses CPU.

### 4.1 Sanitization
- Pydantic validation: max body size, max message length (~256 chars), required fields
- Unicode NFKC normalization (folds fullwidth/styled chars)
- Strip zero-width, control and bidi-override characters; collapse whitespace
- Map homoglyphs (Cyrillic to Latin) and leetspeak (`@`->a, `0`->o, etc.)
- Collapse repeated characters ("fuuuuck" -> "fuuck")
- Keep the **original** text for logs/review; run the model on the normalized copy
- Security: parameterized queries only, API-key auth between game server and API, rate limit the API itself

### 4.2 Spam rules (per-player state)
- Sliding window / token bucket rate limit (e.g. 5 messages per 10 s; stricter for new accounts)
- Duplicate detection via hash of normalized text, plus near-duplicate similarity (~0.9)
- Pattern checks: all-caps ratio, character flooding, emoji/symbol floods, length outliers
- Links and Discord invites blocked/flagged unless allowlisted
- Same text across channels counts as one burst

### 4.3 Harassment tracker (sender-to-target relationship)
- Count messages per (sender, target) pair; many unanswered DMs/mentions is a red flag
- Pile-ons: many distinct senders flagged against one target in a short window
- Escalation: keep last 5-10 messages per pair, watch for rising toxicity
- Player reports as a signal, weighted by the reporter's history (anti-false-report)
- Enforce target controls (block, mute, friends-only) before anything else

### 4.4 Content model
BERT scores each normalized message (truncate to ~128 tokens). Scores feed the player's record and are not a final verdict. Use **per-category thresholds** (e.g. block threat at 0.7, flag insult at 0.9), tuned on the game's real chat.

### 4.5 Decision and behavior score

| Event | Points |
|---|---|
| Rate limit hit | +1 |
| Duplicate/spam | +2 |
| Toxic block | +5 |
| Harassment pattern | +10 |
| Confirmed report | +10 |

Points decay (e.g. -1 per hour). Escalation ladder: allow -> warn -> flag for human review -> temp mute (10 min, 1 hr, 24 hr) -> human decides on ban.

## 5. API contract (draft)

```
POST /moderate
{"text": "...", "player_id": "123", "target_id": "456" (optional), "channel": "global"}

200 {
  "action": "allow" | "flag" | "block" | "shadow_mute",
  "reasons": ["rate_limit", "toxic:0.97", ...],
  "scores": {"toxic": 0.97, "insult": 0.91, "threat": 0.02, ...},
  "player_score": 14
}

GET /health
```

## 6. Redis keys (example)

```
rate:{player}              sorted set of timestamps
dup:{player}:{msg_hash}    counter, TTL 60s
pair:{sender}:{target}     counter, TTL 1h
score:{player}             behavior score (with decay)
```

## 7. Deployment notes

- Slim Python base image; deps: `fastapi`, `uvicorn`, `onnxruntime`, `tokenizers` (avoid full PyTorch in the runtime image)
- Bake model files into the image (no download at startup)
- A few uvicorn workers; cap ONNX Runtime intra-op threads per worker so workers don't fight over cores
- LRU cache for repeated messages ("gg", "noob")
- Log every decision with its reason for moderator review and appeals
- docker-compose with two services: `moderation-api` and `redis`

## 8. Known limitations (state these honestly in the presentation)

- toxic-bert was trained on Jigsaw (online comments): it can misjudge gaming slang and trash talk, and has known bias around identity terms
- English-focused
- Sees one message at a time, with no conversation context (hence the Redis layer)
- Determined harassers evade filters (new accounts, coded language); human review and a report button are needed
- Latency figures (roughly 10-30 ms per short message on CPU with ONNX int8) are **unbenchmarked estimates**; measure on the real EPYC server
- Model details, labels and licenses were recalled from memory: **verify on the Hugging Face model cards** before deploying

## 9. Next steps (suggested order)

1. Project skeleton: `app/` (main.py, normalize.py, spam.py, harassment.py, model.py, scoring.py), `Dockerfile`, `docker-compose.yml`, `requirements.txt`
2. Implement and unit test `normalize.py` (leetspeak, homoglyphs, repeats, NFKC)
3. Export toxic-bert to ONNX + int8 quantize; confirm outputs match the PyTorch model on sample inputs
4. `/moderate` endpoint with model scoring only; add per-category thresholds
5. Add Redis rate limiter, duplicate detection and behavior score with decay
6. Add the harassment tracker (pair counters, pile-on detection)
7. Dockerize, then benchmark latency and throughput on the EPYC VPS
8. Build a small demo script or web page that sends sample chat (clean, toxic, spam burst, harassment sequence) and shows the decisions and reasons
9. Tune thresholds against realistic game-chat samples

## 10. Instructions for the next LLM

- Treat sections 3-7 as agreed decisions. Raise concerns, but don't silently change the stack.
- Prefer small, runnable steps. Start with step 1 and produce working code.
- Keep CPU-only inference and low latency as hard requirements.
- Flag anything uncertain (model names, library versions, API details) rather than guessing.
