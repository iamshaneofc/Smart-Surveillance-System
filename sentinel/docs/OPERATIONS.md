# SENTINEL — Operations Guide (F2 + F3 + F4)

Day-2 operations: running the API, the operator web console, evidence
retention, alert delivery, audit, and logging. Everything here is
self-hosted/local — no cloud, Kubernetes, or managed services.

## 1. Running the API

```bash
cd sentinel
.venv/bin/uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Configuration is environment-only (`SENTINEL_*`, nested via `__`); nothing is
hard-coded per deployment. Important variables:

| Variable | Default | Purpose |
|---|---|---|
| `SENTINEL_ENV` | `dev` | environment label |
| `SENTINEL_API_HOST` / `SENTINEL_API_PORT` | `127.0.0.1` / `8000` | bind address |
| `SENTINEL_DATABASE_URL` | `sqlite:///./var/sentinel.db` | database DSN |
| `SENTINEL_AUTH_MODE` | `disabled` | `disabled` (dev only) \| `api_key` |
| `SENTINEL_AUTH_API_KEYS` | `[]` | JSON list of `{key,user,roles}` |
| `SENTINEL_RATE_LIMIT_PER_MINUTE` | `120` | per-principal sliding window |
| `SENTINEL_LOG_LEVEL` / `SENTINEL_LOG_JSON` | `INFO` / `false` | logging |
| `SENTINEL_EVIDENCE__ROOT` | `./var/evidence` | evidence storage root |

**Production checklist:** `SENTINEL_AUTH_MODE=api_key` with real keys, a
non-SQLite database if multi-process, `SENTINEL_LOG_JSON=true`, and the
retention sweep scheduled (§2). Never log or paste API keys or RTSP
credentials — the code redacts them (`redact_secrets`), do not defeat that.

## 2. Evidence retention

Evidence rows expire per severity (`critical 90d, high 30d, medium 14d, low
3d` — configurable via `SENTINEL_EVIDENCE__RETENTION_DAYS_BY_SEVERITY`).

**Sweep command:**

```bash
# Report only — deletes nothing, always exits 0
.venv/bin/python scripts/run_retention_sweep.py --dry-run

# Real sweep — deletes expired evidence, exits 1 if any deletion failed
.venv/bin/python scripts/run_retention_sweep.py
```

Behaviour guarantees (tested):

- **Order:** storage file first, then the metadata row. If the file delete
  fails, the row is kept, the failure is audited
  (`evidence.retention_failed`) and logged; the sweep continues.
- **Idempotent:** running twice deletes nothing the second time; a missing
  file is recorded as `reason: file_missing` and the row is still removed.
- **Auditable:** every removal writes `evidence.retention_delete` with
  `event_id`, `uri`, `sha256`, `expires_at`, `reason`
  (`expired` | `file_missing`) and actor `system:retention`.
- **Safety:** evidence tied to an *active* event (`new`/`acknowledged`) is
  skipped (counted in `skipped_active`) unless
  `SENTINEL_EVIDENCE__ALLOW_ACTIVE_EVENT_DELETION=true`.

**Scheduling** (daily at 03:15):

- Linux cron: `15 3 * * * cd /opt/sentinel && .venv/bin/python scripts/run_retention_sweep.py >> var/retention.log 2>&1`
- Windows Task Scheduler: `schtasks /Create /TN "SENTINEL retention" /SC DAILY /ST 03:15 /TR "Z:\...\sentinel\.venv\Scripts\python.exe Z:\...\sentinel\scripts\run_retention_sweep.py"`

**Verify:** check the JSON summary (`candidates/deleted/failed/
skipped_active/missing_files`), the exit code, and audit rows:

```sql
SELECT actor, action, resource_id, details FROM audit_log
WHERE action LIKE 'evidence.retention%' ORDER BY id DESC LIMIT 20;
```

The API itself does not sweep on a timer in F2 — the sweep is an explicit
operator action (also runnable from CI).

## 3. Alerts

Alerts are produced by the pipeline after an event is confirmed; delivery is
decoupled so **no alert failure can crash or stall event processing**
(failures are logged, counted, and audited — events still persist).

**Configuration** (`SENTINEL_ALERTS__*`, all optional):

| Variable | Default | Notes |
|---|---|---|
| `SENTINEL_ALERTS__ENABLED` | `false` | pipeline submits alerts only when enabled |
| `SENTINEL_ALERTS__CHANNELS` | `["in_app"]` | only `in_app` (database rows) and `webhook` are accepted; an unknown channel fails fast at startup |
| `SENTINEL_ALERTS__WEBHOOK_URL` | `""` | must start `http://`/`https://` — validated fail-fast at startup |
| `SENTINEL_ALERTS__WEBHOOK_TIMEOUT_SECONDS` | `5.0` | per attempt (1–60) |
| `SENTINEL_ALERTS__WEBHOOK_MAX_RETRIES` | `2` | extra attempts after the first (0–5) |
| `SENTINEL_ALERTS__WEBHOOK_BACKOFF_SECONDS` | `0.5` | exponential backoff base |
| `SENTINEL_ALERTS__COOLDOWN_SECONDS` | `120` | per camera+event_type+channel |
| `SENTINEL_ALERTS__RATE_CAP_PER_MINUTE` | `30` | per camera |
| `SENTINEL_ALERTS__QUEUE_MAX` | `100` | bounded queue; overflow drops (counted) instead of blocking the pipeline |

Webhook payload (secrets never included):

```json
{"alert_id": "alr_...", "event_id": "evt_...", "event_type": "restricted_zone_intrusion",
 "severity": "high", "camera_id": "cam_gate", "timestamp": "2026-10-03T12:00:00+00:00",
 "title": "HIGH restricted_zone_intrusion @ cam_gate",
 "summary": "Track #1 entered Restricted Area and remained inside for 1.9 seconds.",
 "evidence_ids": ["evd_..."], "body": "...", "metadata": {"rule_id": "...", "channel": "webhook"}}
```

**Guarantees (tested):** delivery is idempotent per `event_id` (duplicates
return `skipped/duplicate`); retries are bounded with backoff; the notifier
raises `WebhookDeliveryError` with the message passed through
`redact_secrets`; a dead webhook never fails the pipeline.

**Verify:**

```bash
curl -s "http://127.0.0.1:8000/api/v1/alerts?limit=20" -H "X-API-Key: operator-key"
```

and grep logs for `alert send failed` / `alert dispatcher started`.

## 4. Audit trail

Every meaningful mutation writes a row to `audit_log`
(`actor, action, resource_type, resource_id, details, ts`):

| Action | Actor |
|---|---|
| `camera.create` / `camera.update` / `camera.delete` | caller (`admin1`, …) or `system` |
| `zone.create` / `zone.update` / `zone.delete` | caller |
| `rule.create` / `rule.update` / `rule.delete` | caller |
| `evidence.download` | caller |
| `evidence.retention_delete` / `evidence.retention_failed` | `system:retention` |

Denied requests (403) write **no** audit rows (tested). Query:

```sql
SELECT ts, actor, action, resource_type, resource_id
FROM audit_log ORDER BY id DESC LIMIT 50;
```

`audit:read` permission exists for a future audit endpoint; in F2 access is
via the database directly.

## 5. Logging

- Structured via `packages/common/logging.py`; optional JSON
  (`SENTINEL_LOG_JSON=true`) with `ts/level/logger/msg/request_id/camera_id`
  plus any `extra` fields.
- Request correlation: each request carries a `request_id` propagated into
  logs; event errors carry `camera_id`.
- **Redaction:** URL credentials and API-key-shaped values are stripped by
  `redact_secrets` in camera/alert/retention/API error paths — including the
  `internal_error` log from `unhandled_error_handler`
  (`log.exception`, traceback included, secrets redacted).
- Unhandled API exceptions return a generic
  `{"error":{"code":"internal_error",...}}` (no stack traces to clients) and
  are logged server-side with traceback.
- Detector/evaluation logs: retention logs deletions, alert logs failures,
  evaluation logs `evaluation skipped` with a reason — all verified by
  `tests/test_f2_observability.py`.

## 6. Health & regression checks

```bash
curl -s http://127.0.0.1:8000/api/v1/system/health   # bus, db, pipeline
.venv/bin/python -m pytest -q                        # full suite
.venv/bin/python -m alembic check                    # schema == models
.venv/bin/python scripts/run_live_acceptance.py      # 17/17 live checks
```

The live acceptance run must stay **17/17**; it never claims accuracy
numbers (see `docs/MODEL_REGISTRY.md`).

## 7. Operator web console (F3)

### Starting

```bash
# terminal 1 — backend (from sentinel/)
.venv/Scripts/uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

# terminal 2 — frontend (from sentinel/apps/web/)
npm install
npm run dev            # http://127.0.0.1:5173, proxies /api -> :8000
```

### Frontend environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE` | `/api/v1` | API base path (served through the dev proxy) |
| `VITE_API_PROXY_TARGET` | `http://127.0.0.1:8000` | backend target for the dev proxy |

No backend host is hard-coded in the UI code.

### Authentication (development modes)

- `SENTINEL_AUTH_MODE=disabled` (default): `/auth/me` reports a dev admin —
  no key needed. Development only.
- `SENTINEL_AUTH_MODE=api_key`: the console shows a login gate; the key is
  stored in `localStorage` (`sentinel.apiKey`) and sent as `X-API-Key` on
  every request. Bad keys surface the backend error (401/429).

Frontend hiding of controls is convenience only — the backend remains the
authority (401/403/429 server-side, verified by tests).

### Pages and operator roles

| Page | Route | Minimum permission |
|---|---|---|
| Command Center (dashboard) | `/` | `events:read` |
| Camera wall + management | `/cameras` | `cameras:read`; create/edit/delete need `cameras:manage` |
| Camera detail (preview, zones) | `/cameras/:id` | `cameras:read`; zone drawing needs `zones:manage` |
| Event feed / detail | `/events`, `/events/:id` | `events:read`; acknowledge/dismiss need `events:ack` / `events:dismiss` |
| Alert center | `/alerts` | `alerts:read` |
| Rule configuration | `/rules` | `rules:read`; mutations need `rules:manage` |
| Health | `/health` | `health:read` |

Roles: **viewer** (read-only + evidence view), **operator** (event actions,
rule/zone reading), **admin** (everything, including `evidence:export`
download and model/user/system administration).

### Evidence access

Snapshots/clips render inline for any principal with `evidence:read`;
**download/export requires `evidence:export` (admin)** — the UI hides the
buttons and the server rejects with 403 anyway. Expired/unavailable
evidence shows a clear state, never a broken page.

### Live video limitations (read this before expecting streams)

- Browsers cannot consume arbitrary RTSP directly; there is **no media
  server** in F3 (no HLS/WebRTC).
- Camera tiles show `PREVIEW` (a development frame from
  `GET /cameras/{id}/preview.jpg`, refreshed by the UI) or `UNAVAILABLE`.
  They are never labelled "live video".
- The header `LIVE` badge refers only to the **event stream**
  (`GET /api/v1/stream` SSE); if it drops, the console falls back to
  `POLLING · 10s`, then shows `OFFLINE`.
- Future path (not implemented): RTSP → media gateway → HLS/WebRTC →
  browser. The preview abstraction is designed to be replaceable.

### Known limitations (console)

- Single-window, desktop-first (usable on tablet/phone via a slide-in nav);
  no mobile app.
- Zone drawing works on the preview frame or an approximate fixed canvas
  (labelled) when no frame size is known; polygons are stored normalized
  0–1.
- `stream_url` is never displayed (write-only in the API); entering a new
  URL is supported, reading the old one is not.
- Real-time transport covers events/alerts/camera health; zone/rule/camera
  edits appear after navigation or their own reload, not via the stream.
- No SSO/OIDC; API-key or disabled-auth modes only (backend limits, not UI).

## 8. Detector operations (F4)

### Acquiring and verifying model weights

Weights are **never committed and never downloaded at runtime**. A missing
artifact is a deterministic configuration error that prints the command to run.

```bash
# from sentinel/
.\.venv\Scripts\python.exe scripts\download_models.py            # download + sha256 verify
.\.venv\Scripts\python.exe scripts\download_models.py --check    # verify only (exit 1 if missing/bad)
```

Pinned artifact (`scripts/weights.lock.json`): `yolox_tiny.onnx`,
20,219,662 B, sha256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`.
See `models/README.md` for provenance and `DATA_LICENSES.md` W1 for license
status (REVIEW REQUIRED before any commercial claim).

### Selecting a detector profile

```bash
# default stays HOG (development only, never evaluated)
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --port 8000

# opt in to the YOLOX-Tiny candidate (status: candidate, unevaluated)
$env:SENTINEL_PIPELINE__DETECTOR_PROFILE = "yolox"
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --port 8000
```

- Profiles: `hog` (default), `stub`, `null`, `yolox` — unknown names fail at
  startup listing the registered ones.
- `SENTINEL_PIPELINE__DETECTOR_OPTIONS` (JSON) passes adapter options, e.g.
  `{"confidence_threshold": 0.4, "weights_path": "models/yolox_tiny.onnx"}`.
- If yolox is selected and the weights are missing, the pipeline fails to
  start with the acquisition command; **the API keeps running** (logged
  `pipeline failed to start - API continues without it`) — fix weights and
  restart. Camera health then reports the camera as never-started rather than
  pretending to be healthy.

### Failure behavior in a running pipeline

- Per-frame detector failure (corrupt frame, inference error) →
  `detector_errors` increments, camera `ai_status` becomes `degraded`, the
  worker keeps running and `ai_status` returns to `healthy` when the detector
  recovers. A detector failure is **never** shown as "no detections".
- Missing/invalid configuration at startup → clear `DetectorError` (weights
  path + download command, or the exact invalid option).
- Health distinction: `offline`/`retrying` = camera problem;
  `degraded` + `ai_status=degraded` = camera feeding but AI failing
  (`GET /api/v1/cameras/health/summary`).

### Benchmarks (latency only — never accuracy)

```bash
.\.venv\Scripts\python.exe scripts\benchmark_pipeline.py --source <video.mp4> --profile yolox --json var/benchmarks\run.json
.\.venv\Scripts\python.exe scripts\benchmark_pipeline.py --source synthetic --frames 200 --warmup 3
```

The JSON records hardware, software versions, warmup policy, batch size,
input resolution, confidence threshold and model status. Numbers are
latency/throughput for this machine only. Reference run (internal clip,
219 frames): detector p50 34.9 ms / p95 48.8 ms / p99 74.6 ms, 24.8 fps,
0 errors — see `DETECTOR_SELECTION.md` section 12.

### Evaluation (blocked until licensed data exists)

```bash
.\.venv\Scripts\python.exe evaluation\validate_manifest.py <manifest.json>   # 8 checks, exit 0/1
.\.venv\Scripts\python.exe evaluation\runners\run_evaluation.py <manifest.json>
.\.venv\Scripts\python.exe evaluation\runners\run_threshold_sweep.py <manifest.json>
```

Without a valid licensed dataset all runners return
`EVALUATION DATASET NOT AVAILABLE` with null metrics and
`operational_threshold: NOT EVALUATED` (exit code 3) — this is the expected
state today (`DETECTOR_SELECTION.md` section 10). Never hand-edit a report to
look evaluated.

### End-to-end chain verification

```bash
.\.venv\Scripts\python.exe scripts\run_f4_event_verification.py   # 28 checks, exit 0
```

Runs the production wiring twice over the internal development clip: the stock
factory pack (honest non-confirmation of short conditions) and the documented
`f4-verification.yaml` pack (full rule → event → evidence → alert → API chain
with `yolox-tiny:0.1.1rc0` provenance). Report:
`var/benchmarks/f4-event-verification.json`. Development footage only — never
presented as an accuracy or operational-performance result.

### Known limitations (F4)

- `EvidenceService` keeps **one evidence session per camera**: when two events
  confirm near-simultaneously, the earlier event receives no evidence rows
  (observed in F4-O; pre-existing design, candidate F5 item).
- The IOU tracker is a development tracker (`MODEL_REGISTRY`/docs); no
  production tracker selection has happened.
- `f4-verification.yaml` shortens confirmation to 0.6 s **only** to exercise
  the chain on a 7-second development clip; do not deploy it as a site pack.
