# Game chat moderation API (demo)

A moderation service for in-game text chat. It handles **toxic content**,
**spam** and **harassment**, and returns a decision the game server can act
on directly.

CPU-only, low latency, no GPU. The classifier is a BERT-base model exported
to ONNX and dynamically quantized to int8 (105 MB), served by ONNX Runtime.
State lives in Redis with TTLs on every key.

> Built from `HANDOFF.md`. The design decisions in that document are treated
> as agreed; where reality forced a change or left something unverified, it is
> called out explicitly in [Deviations and open questions](#deviations-and-open-questions).

---

## The pipeline

```
Request -> 1. Sanitize -> 2. Spam rules -> 3. Harassment tracker -> 4. BERT -> 5. Decision + player score
```

Cheap checks run first, so most abuse is caught before the classifier touches
a CPU cycle. When a cheap check has already decided the outcome, the model is
skipped entirely (`MOD_SKIP_MODEL_ON_SPAM=true`).

| Stage | Module | What it does |
|---|---|---|
| 1. Sanitize | `app/normalize.py` | NFKC folding, zero-width/bidi stripping, homoglyph and leetspeak mapping, repeated-char collapse. Keeps the original for logs. |
| 2. Spam | `app/spam.py` | Sliding-window rate limit, exact and near-duplicate detection, caps/character/symbol/length floods, link and invite detection. |
| 3. Harassment | `app/harassment.py` | Unanswered targeting, rising toxicity within a pair, pile-on detection, weighted player reports. |
| 4. BERT | `app/model.py` | ONNX Runtime int8 scoring, padded into fixed buckets, with an LRU cache. |
| 5. Decision | `app/service.py`, `app/scoring.py` | Per-category thresholds, behaviour score with time decay, escalation ladder. |

---

## Quick start

### Locally, no Docker

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt

.venv/bin/python -m pytest tests/ -q        # 96 tests
.venv/bin/python demo/demo_chat.py           # scripted demonstration
```

### With Docker

```bash
cp .env.example .env
# Both secrets are required; the compose file refuses to start without them.
printf 'MOD_API_KEY=%s\nMOD_REDIS_PASSWORD=%s\n' \
  "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" >> .env
docker compose up --build
curl localhost:8000/health
```

Then open <http://localhost:8000/demo> for the live demo page.

For local development without Docker, the service refuses to start on the
well-known development key. Either set `MOD_API_KEY` to something real, or set
`MOD_ALLOW_INSECURE_DEFAULT_KEY=true` for local use only.

### Rebuilding the model

The quantized model is not committed (105 MB, fully reproducible):

```bash
uv venv --python 3.12 .venv-export
uv pip install --python .venv-export/bin/python torch==2.15.0 \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-export/bin/python -r requirements-export.txt
.venv-export/bin/python tools/export_model.py
```

The export verifies itself: the float ONNX graph must match PyTorch to within
1e-3 logits, and the int8 graph must not change a single decision at either
service threshold.

---

## API

### `POST /moderate`

Requires the `X-API-Key` header.

```json
{
  "text": "you are a disgusting idiot",
  "player_id": "123",
  "target_id": "456",
  "channel": "global",
  "account_age_s": 86400,
  "is_report": false
}
```

| Field | Required | Meaning |
|---|---|---|
| `text` | yes | The raw message, up to 256 characters. |
| `player_id` | yes | The sender. `[A-Za-z0-9_-]{1,64}` — it is interpolated into Redis keys. |
| `target_id` | no | Who the message is aimed at (DM recipient, mentioned player). |
| `channel` | no | Defaults to `global`. |
| `account_age_s` | no | Sender account age; newer accounts get a tighter rate limit. |
| `is_report` | no | `true` when `player_id` is reporting `target_id`. |

```json
{
  "action": "flag",
  "reasons": ["toxic:0.99", "insult:0.97"],
  "scores": {"toxic": 0.9903, "insult": 0.966, "threat": 0.0017},
  "player_score": 2,
  "target_score": null,
  "escalation": "none",
  "muted_for_s": null,
  "latency_ms": 3.6
}
```

| Action | What the game server should do |
|---|---|
| `allow` | Deliver the message. |
| `flag` | Deliver it, and queue it for human review. |
| `block` | Reject it and tell the sender why. |
| `shadow_mute` | Deliver it to the sender only; nobody else sees it. |

`reasons` is the evidence, machine-readable: `rate_limit`, `duplicate`,
`near_duplicate`, `link`, `discord_invite`, `external_invite`, `caps_flood`,
`char_flood`, `symbol_flood`, `length_flood`, `unanswered_targeting`,
`escalating_toxicity`, `pile_on`, `report`, `confirmed_report`, plus one
entry per classifier label that crossed a threshold (`threat:0.82`).

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "model_id": "unitary/toxic-bert", "redis_ok": true, "version": "0.1.0"}
```

`status` is `degraded` when Redis is unreachable. The service keeps working:
the stateful heuristics are skipped, the classifier still runs, and the
response carries a `state_unavailable` reason.

### `GET /demo`, `GET /demo/config`

A small page for live demonstrations, and the settings it needs. The config
endpoint never exposes the API key.

---

## Behaviour score and escalation

Every message adds points to the sender's score. The score decays by one
point per hour (`MOD_SCORE_DECAY_INTERVAL_S`), so an old isolated offence
stops counting.

| Event | Points |
|---|---|
| Rate limit hit | +1 |
| Duplicate / spam | +2 |
| Toxic flag (soft category) | +2 |
| Toxic block (hard category) | +5 |
| Harassment pattern | +10 |
| Confirmed report | +10 |
| Unconfirmed report | +0 |

The score drives the ladder:

```
allow -> warn -> flag for review -> temp mute (10 min, 1 hr, 24 hr) -> human decides on ban
```

A mute lasts for one incident. The player has to fall back below the mute
band and cross it again before the next, longer mute applies.

### Player reports

A report is a *claim*, not evidence, so it only counts when something
corroborates it:

* there must be a real exchange between the two players — a report cannot be
  aimed at a stranger;
* the **target's own** messages in that exchange must have been scored as
  toxic (`MOD_FLAG_THRESHOLD`);
* the reporter's own history scales the weight, so a serial false-reporter
  loses influence;
* one reporter may contribute at most `MOD_REPORT_POINTS_CAP` points to one
  target per `MOD_REPORT_POINTS_WINDOW_S`.

Confirmation deliberately does not depend on the target's score: a score the
reporter can move is not evidence. An uncorroborated report is still recorded
and visible to a human reviewer — it just cannot move a player up the ladder
on its own, because that is how a report becomes a weapon.

---

## Configuration

Everything is an environment variable with the `MOD_` prefix; see
`.env.example` for the full list with defaults. The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `MOD_API_KEY` | dev placeholder | The service logs a warning while the placeholder is in use. |
| `MOD_REQUIRE_API_KEY` | `true` | |
| `MOD_REDIS_URL` | `redis://localhost:6379/0` | |
| `MOD_ONNX_THREADS` | `2` | Keep `workers * threads` under the core count. |
| `MOD_BLOCK_THRESHOLD` | `0.70` | Hard categories: `threat`, `identity_hate`, `severe_toxic`. |
| `MOD_FLAG_THRESHOLD` | `0.90` | Soft categories: `toxic`, `insult`, `obscene`. |
| `MOD_RATE_LIMIT_COUNT` / `_WINDOW_S` | `5` / `10` | 5 messages per 10 seconds. |
| `MOD_NEW_ACCOUNT_RATE_LIMIT` | `3` | For accounts under `MOD_NEW_ACCOUNT_AGE_S` (1 hour). |
| `MOD_SKIP_MODEL_ON_SPAM` | `true` | Skip the classifier when a cheap check already decided. |

### Redis keys

Every key has a TTL; nothing durable is stored.

```
rate:{player}              sorted set of send timestamps       (sliding window)
dup:{player}:{hash}        counter of identical messages        (60 s)
recent:{player}            list of recent normalized texts      (capped, 300 s)
pair:{sender}:{target}     directional message counter         (3600 s)
pairmsgs:{a}:{b}           shared conversation history          (capped, 3600 s)
pileon:{target}            distinct senders in the window        (300 s)
score:{player}             hash: points, updated_at, mute_tier, mute_duration
mute:{player}              mute count, drives the mute length
reports_made:{reporter}    report count, drives reporter trust
reports_against:{target}   report count against a player
reportpoints:{a}:{b}       points one reporter has added to one target
apilimit:{key}             API-level rate limit counter
```

### Decision logging

Every decision is written to stdout as one JSON object, with the original
text, the reasons, the scores and the timing. That stream is the audit trail
for moderator review and player appeals:

```json
{"ts":"2026-11-14T22:23:09Z","level":"INFO","logger":"app.service","message":"moderation decision","decision":{"player_id":"dave","text":"I will find you and break your legs","action":"block","reasons":["threat:0.82","escalation:warn"],"player_score":5,"escalation":"warn","latency_ms":3.59}}
```

---

## Security

The game server holds an API key; the chat content inside a request is
attacker-influenced. The controls that follow from that:

| Control | Where |
|---|---|
| API key checked in constant time, before the body is read | `app/security.py` |
| Request body capped at 4 KB *while streaming*, so a chunked request cannot exceed it | `app/security.py` |
| Duplicate `X-API-Key` headers rejected | `app/security.py` |
| The service refuses to start on the public development key | `app/config.py` |
| Per-client API rate limit, with a local fallback when Redis is down | `app/ratelimit.py` |
| Player ids and channels restricted to `[A-Za-z0-9_-]`, so they cannot escape the Redis key namespace | `app/schemas.py` |
| Reports only count when corroborated by the target's own behaviour | `app/harassment.py` |
| Redis requires a password, is not published to the host, and evicts only TTL'd keys | `docker-compose.yml` |
| Container runs as a non-root user with no capabilities, a read-only filesystem and a memory limit | `Dockerfile`, `docker-compose.yml` |

Two decisions worth stating plainly:

**The body limit is enforced in raw ASGI middleware, not in a FastAPI
dependency.** FastAPI resolves a request body *before* it runs route
dependencies, so a dependency-level limit would let an unauthenticated caller
make the service buffer an arbitrarily large body first. Measured before the
fix: a 100 MB chunked body from a caller with the wrong key cost **315 MB** of
heap before the 401. After: **0 MB**, because the key is checked before any
body is read.

**The rate limiter does not fail open.** When Redis is unreachable the service
falls back to a local in-process counter rather than skipping the limit: the
classifier is the expensive resource, so "no limit" is the wrong failure
mode.

### Decision log and privacy

Every decision is logged with the original message text, which is what makes
the log usable for moderator review and player appeals. That also means the
log stream can contain whatever players typed — personal details, handles,
anything. Two consequences to own before deploying:

* `MOD_LOG_MESSAGE_TEXT=false` omits the raw text from the log; the hash and
  the decision remain.
* Nobody has set a retention period or an access policy for this log yet,
  because that depends on the operator. It needs one.

### Still open

* `pip-audit` was not run (no network access for the advisory database). Run
  it in CI before this reaches the VPS.
* The base image is not digest-pinned, and `numpy` is the one dependency with
  a version range rather than an exact pin.
* `player_id` is attacker-influenced, so rotating it rotates the per-player
  rate limit. The per-client limit in front of it is the mitigation, but a
  game-side session identity would be better.
* The number of distinct player ids the service will track is unbounded. Every
  id creates up to ten keys, two of which live for a day and a week.

---

## Model

`unitary/toxic-bert` — BERT-base, 110M parameters, the model behind Detoxify.
Multi-label: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`,
`identity_hate`. Licence: Apache-2.0.

Exported to ONNX at opset 18 and dynamically quantized to int8 (105 MB). The
runtime image ships `onnxruntime` and `tokenizers` only — no PyTorch, no
Transformers.

Measured agreement between the int8 graph and the PyTorch reference on the
verification set: max probability difference **0.020**, and **zero** label
threshold flips at either threshold the service uses.

The verification is deliberately built the way the service is:

* it loads the **shipped** `models/tokenizer.json`, not a freshly downloaded
  one, and lets the tokenizer do the truncating — truncating the token ids by
  hand drops the trailing `[SEP]`, which is a different input and changes the
  scores;
* it scores one message at a time, the way the API serves it, because dynamic
  int8 quantization derives its activation scales from the tensor it sees;
* it checks the thresholds from `app/config.py` rather than copies of them;
* the artifact is only moved into `models/` once it has passed, so a failed
  export cannot be shipped by a later `docker build`.

### Latency

On the development machine (AMD Ryzen 7 9800X3D, 2 ORT threads), model-only,
short chat messages:

| Measurement | Value |
|---|---|
| Cold start (first inference) | 3.8 ms |
| Warm, cache miss | p50 2.55 ms, p90 2.65 ms, p99 2.76 ms |
| Warm, cache hit | ~0.01 ms |
| Full pipeline, end to end | ~3.6 ms |

**These are not the target numbers.** The deployment target is an AMD EPYC
VPS, and the handoff's 10-30 ms estimate was explicitly unbenchmarked. Run
`tools/benchmark.py` on the EPYC box before quoting a figure:

```bash
.venv/bin/python tools/benchmark.py --messages 1000
.venv/bin/python tools/benchmark.py --url http://localhost:8000 --messages 500
```

---

## Testing it

Four levels, from fastest to most complete.

### 1. The test suite

```bash
.venv/bin/python -m pytest tests/ -q       # 125 tests, ~7 s
.venv/bin/ruff check .
```

Runs entirely in-process against `fakeredis`, and loads the real quantized
model, so it covers the export as well as the service.

### 2. The scripted demo — no server, no Redis

```bash
.venv/bin/python demo/demo_chat.py
```

Prints seven scenarios end to end: clean chat, a toxic message, a threat, a
spam burst, a harassment sequence, an evasion attempt and a player report,
each with the action, the reasons and the classifier scores. This is the
quickest way to show the pipeline doing its thing — it needs nothing but the
model file.

### 3. The live service and the demo page

```bash
# Redis is needed for spam, harassment and the behaviour score.
docker compose up --build

# Or, without Docker, run Redis yourself and:
MOD_API_KEY=some-long-dev-key \
MOD_REDIS_URL=redis://localhost:6379/0 \
  .venv/bin/uvicorn app.main:app --port 8000
```

Then open <http://localhost:8000/demo>. The page has one-click samples
(clean, toxic, threat, link, caps flood, emoji flood, harassment, evasion)
and shows the decision, the reasons, the per-label scores and the latency.

Two things to know:

* The service **refuses to start** on the well-known development key. Set
  `MOD_API_KEY` to at least 16 characters, or
  `MOD_ALLOW_INSECURE_DEFAULT_KEY=true` for local work.
* Without Redis it still runs and still classifies, but the spam burst,
  harassment and behaviour score do not fire, and the response says
  `state_unavailable`. The stateless checks — caps floods, links, Discord
  invites — keep working either way.

### 4. A deployed instance

```bash
.venv/bin/python tools/smoke_test.py --url http://localhost:8000 \
    --api-key "$MOD_API_KEY"
```

Ten checks against a running service over HTTP: health, authentication, the
main decisions, that a threat carrying a link is still blocked, that the
behaviour score grows, that a report about a stranger moves nothing, and
that an oversized body is refused. Exits non-zero on failure, so it works as
a deployment gate. Checks that need Redis are *skipped*, not failed, when
Redis is unreachable.

### 5. Latency

```bash
.venv/bin/python tools/benchmark.py --messages 1000
.venv/bin/python tools/benchmark.py --url http://localhost:8000 --messages 500
```

### Suggested 15-minute run-of-show

1. `pytest -q` — "the whole thing is tested; here's the count."
2. `demo/demo_chat.py` — walk through the scenarios; point at the `reasons`
   line to show the decision is explainable, not a black box.
3. Open `/demo` — click the **Threat** and **Spam burst** samples; show the
   score bars and the latency.
4. `tools/smoke_test.py` — "this is what we run against the box before
   pointing the game server at it."
5. Open `app/service.py` — the pipeline is one readable function.

### Prerequisites

The model file is not committed (105 MB). If `models/model_int8.onnx` is
missing, regenerate it:

```bash
uv venv --python 3.12 .venv-export
uv pip install --python .venv-export/bin/python torch==2.15.0 \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-export/bin/python -r requirements-export.txt
.venv-export/bin/python tools/export_model.py
```

The export verifies itself and refuses to publish an artifact that changes a
decision.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 96 tests
.venv/bin/ruff check .
```

The suite runs against `fakeredis`, so no Redis server is needed. The
end-to-end tests load the real quantized model, so they cover the export too.

---

## Deviations and open questions

Everything the handoff left uncertain, and everything that changed.

### Verified against upstream

- `unitary/toxic-bert`: label list, order and `multi_label_classification`
  problem type confirmed from `config.json`; licence Apache-2.0.
- `martin-ha/toxic-comment-model`: confirmed as DistilBERT with a
  **single-label** head (`non-toxic`, `toxic`) — not the six-label multi-label
  head the handoff assumed, so it is not a drop-in fallback. **Its model card
  states no licence.** Do not ship it until that is resolved.

### Changed, and why

1. **ONNX opset 18, not 13.** BERT uses `scaled_dot_product_attention`, which
   cannot be represented below opset 14; 18 is what `optimum` recommends.
2. **Two dependency sets.** `optimum-onnx` pins `transformers<4.58` while
   Transformers 5.x is current. The export tool therefore lives in its own
   virtualenv (`.venv-export`) and the runtime image carries no ML framework
   at all.
3. **No Lua.** Atomicity comes from redis-py `MULTI`/`EXEC` pipelines and
   `WATCH` for the score's read-modify-write. This keeps a Lua interpreter out
   of the runtime and lets the whole suite run against `fakeredis`.
4. **`is_report` added to the request.** The handoff's pipeline needs a report
   signal (§4.3) but the draft contract had no field for one.
5. **`target_score`, `escalation`, `muted_for_s` and `latency_ms` added to the
   response.** The demo and the moderation log both need them.
6. **Player IDs are restricted to `[A-Za-z0-9_-]{1,64}`.** They are
   interpolated into Redis keys, so an unrestricted ID could escape the key
   namespace. This is an input-validation rule, not a stack change.
7. **A toxic *flag* adds +2 points.** The handoff's table lists "Toxic block
   +5"; §4.4 also says classifier scores feed the player's record. Flagging
   with no consequence at all would make the flag threshold inert.
8. **An unconfirmed report counts for +5, a confirmed one for +10.** The
   table lists "Confirmed report +10"; an unconfirmed report still contributes
   but should not carry the full weight.
9. **Pile-on is flagged, not muted.** A pile-on is a property of the *target*,
   not of any one sender, so punishing a participant would be wrong. It
   contributes no points against the sender.
10. **Links and invites are flagged; floods and duplicates are shadow-muted.**
    A single link is worth review, not a mute. Spam bursts get the shadow mute
    the handoff specifies.
11. **Soft and hard categories.** The handoff's examples imply one threshold
    per category ("block threat at 0.7, flag insult at 0.9"). Implemented as
    two groups: hard categories block at 0.70, soft categories only flag at
    0.90.
12. **A report only counts when corroborated.** The handoff asked for reports
    "weighted by the reporter's history (anti-false-report)". Taken alone that
    is not enough: without corroboration a player can report a stranger and
    drive them up the ladder. The report must also be backed by the target's
    own scored messages, and one reporter may only add a bounded amount to
    one target. See [Player reports](#player-reports).
13. **Authentication moved into raw ASGI middleware.** FastAPI parses the body
    before it runs dependencies, so the API key and the body limit have to
    happen ahead of routing to bound what an unauthenticated caller can cost.
14. **The development API key is no longer accepted by default.** The service
    refuses to start on it. `MOD_ALLOW_INSECURE_DEFAULT_KEY=true` is the
    local-development escape hatch.
15. **The compose file requires both secrets.** No fallback values, so a
    missing `.env` fails the deployment instead of silently downgrading it.
16. **The classifier is only skipped when delivery is already prevented.**
    The handoff's ordering is kept, but skipping the model on *any* cheap
    verdict would mean a threat that also carried a link was never classified
    and so never blocked. It is skipped only when the cheap check has already
    shadow-muted the message.
17. **Reports are not chat.** A report is not counted as a duplicate or a
    burst, and it does not create an exchange between the two players.
    Otherwise a victim filing several reports would be muted for "spam" and
    their own score would climb.
18. **The API rate limit is a fixed window, per client.** One shared bucket
    keyed by the API key let one caller spend everyone's budget, and it had
    no client dimension to stop a rotating `player_id` bypassing the
    per-player limit.

### Not verified here

- **The Docker image was never built or run.** Docker is not installed on the
  development machine and there is no `sudo`, so `Dockerfile` and
  `docker-compose.yml` are written but unexercised. Build them on the VPS.
- **No real Redis was run.** The suite uses `fakeredis`, which implements
  every command the service uses, but the compose file's Redis configuration
  (password, `volatile-lru`, healthcheck) is untested.
- **No latency measurement on the EPYC target.** See above.
- **Thresholds are untuned against real game chat.** The handoff's §9 step 9
  asks for this; it needs a corpus of actual messages from the game.
- **`martin-ha/toxic-comment-model` is not wired up as a runtime fallback.**
  The service loads exactly one model. Adding a second would mean a second
  ONNX session and a second warm-up, and the licence is unclear.
- **`pip-audit` was not run.** No network access to the advisory database.

### Known limitations of the approach

- toxic-bert was trained on Jigsaw (online comments). It misjudges gaming
  slang and trash talk, and has known bias around identity terms.
- English-focused.
- It sees one message at a time, with no conversation context — which is why
  the Redis layer exists.
- Determined harassers evade filters (fresh accounts, coded language). Human
  review and a report button are still needed.
