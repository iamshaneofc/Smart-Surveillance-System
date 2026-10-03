# SENTINEL — Architecture

**Status: FOUNDATION PHASE. This document defines the target architecture and the subset implemented so far. SENTINEL is NOT production-ready.**

SENTINEL — AI-Powered Video Surveillance & Security Intelligence Platform.

## 1. Principles

1. **Continuous monitoring + event-driven evidence.** Ordinary frames are processed transiently and discarded. Evidence is only persisted when a configured rule fires.
2. **Analytics isolation.** AI inference failure must never break camera ingestion, recording buffers, or stream management. This is the single most important boundary in the system.
3. **Tracks before rules.** Rules operate on tracked objects with persistent IDs over time — never on isolated bounding boxes.
4. **Explainable events, no universal "suspicion score".** Every event carries the explicit conditions that produced it (zone, schedule, dwell, confidence…).
5. **Configuration over code.** Industries (factory, hospital, retail…) are expressed as rule packs, never as separate codebases or hard-coded branches in the engine.
6. **Privacy by design.** On-prem capable, raw video never leaves the site by default, RBAC + audit logging from day one, no face recognition as a default capability.
7. **Honesty about metrics.** No accuracy is claimed until it comes from the evaluation framework with video-level splits.

## 2. Layered architecture

```
CAMERA / VMS (RTSP, ONVIF later, files/webcams for dev)
        ↓
STREAM INGESTION        — connection lifecycle, health, reconnect, heartbeat
        ↓
VIDEO PIPELINE          — sampling, dual streams (detect vs evidence), queues, backpressure
        ↓
OBJECT DETECTION        — pluggable Detector adapters (profile-selected)
        ↓
OBJECT TRACKING         — persistent track IDs (ByteTrack/BoT-SORT later, baseline IOU now)
        ↓
TEMPORAL / ACTIVITY     — clip-window analyzers (future: VideoMAE, pose, VLM verifier)
        ↓
RULE ENGINE             — spatial (zones/lines) + temporal (dwell/confirm) + schedule predicates
        ↓
EVENT ENGINE            — confirmation, dedup, cooldown, severity, state machine
        ↓
EVIDENCE CAPTURE        — pre/post ring buffer → clip + snapshots + metadata + hash
        ↓
EVENT STORE             — PostgreSQL (events, evidence, health, config, audit)
        ↓
ALERT ENGINE            — routing, cooldown, escalation, acknowledgement
        ↓
API (FastAPI)           — REST /api/v1, RBAC, structured errors, versioning
        ↓
WEB UI (React/TS/Tailwind/shadcn — deferred)
```

### Layer responsibilities

| Layer | Owns | Does NOT own |
|---|---|---|
| Stream ingestion | connect/reconnect/backoff, frame delivery, per-camera health (FPS, drops, latency, heartbeat) | any AI, any persistence of ordinary frames |
| Video pipeline | frame sampling to `detection_fps`, detect vs evidence stream split, queues, backpressure, degradation | business rules |
| Detection | `Detector` interface, model profiles, class taxonomy mapping, model metadata/versioning | tracking, events |
| Tracking | track lifecycle, IDs, trajectories, dwell inputs | zone semantics, event creation |
| Temporal analysis | clip-window candidate signals (fight, fall, anomaly) | final event decision |
| Rule engine | zone/line/schedule predicates over tracks, explainable condition lists | evidence storage |
| Event engine | confirm → dedup → cooldown → severity → event states | notifications |
| Evidence capture | pre/post buffers, clip/snapshot assembly, hashing, retention expiry | alerting |
| Alert engine | channels, routing, storm control, escalation, ack | event semantics |
| API | authN/authZ, schemas, pagination, versioning, audit | inference |
| UI | operator workflows | direct model calls |

## 3. Service / process boundaries

Foundation phase runs abstractions inside one API process for simplicity. Target topology:

- **`apps/api`** — REST API, event store, reads/writes config, serves UI (later). Never runs inference in-process.
- **camera service** (`services/camera`) — per-camera worker processes/threads owning ingestion + pipeline. Independent restart.
- **inference workers** (`services/inference`, `services/tracking`) — GPU/CPU workers consuming detect-stream frames.
- **evidence + alert workers** (`services/evidence`, `services/alerts`) — triggered by the event engine via the bus.

Frame data **stays in-process** between ingestion, detection, tracking and buffering. The bus carries only small messages: events, health, alerts, control. Pushing raw frames over the bus would flood it and couple services; it is explicitly out of scope.

### Event bus topics (design)

| Topic | Producer | Consumer(s) |
|---|---|---|
| `events.created` / `events.updated` | event engine | evidence, alerts, API (UI push) |
| `camera.health` | camera workers | API health, alert engine |
| `alerts.dispatch` | alert engine | notifiers |
| `system.status` | all | API health |

**Bus choice: Redis Streams.** Rationale in §14.

## 4. Camera ingestion

Per-camera configuration (entity `Camera`, schema `CameraConfig`):

`camera_id, name, location, stream_url, source_type (rtsp/file/webcam/synthetic), enabled, detection_enabled, recording_enabled, detection_fps, resolution, timezone, retention_policy, model_profile, rule_profile, site`

Connection state machine:

```
IDLE → CONNECTING → STREAMING ⇄ RECONNECTING → OFFLINE → STOPPED
                       ↑ backoff(exponential, capped) ↑
```

Health tracked per camera: FPS (EMA), frame drops (frame_id gaps), read latency, reconnect count, last frame timestamp, error text, state — persisted to `CameraHealth` and surfaced as `HEALTHY | DEGRADED | OFFLINE | ERROR`.

A camera source is never assumed healthy: every `read()` can fail, and workers must handle it with backoff and graceful shutdown (signal → flush → close).

Sources: `RTSPSource` / `FileSource` / `WebcamSource` (OpenCV, optional dependency), `SyntheticSource` (dependency-free, used by tests and demos). ONVIF discovery is a future adapter behind the same `CameraSource` interface — **no ONVIF compatibility is claimed until tested.**

## 5. Video pipeline

- Two logical streams per camera:
  - **Detection stream** — substream/low-res (e.g. 640–1280 px), sampled to `detection_fps` (default 5). This is what models consume.
  - **Evidence stream** — main/high-res frames for ring-buffer capture when an event fires. Falls back to detection frames when no main stream exists.
- Frame queues are bounded; when a consumer falls behind, the oldest detect-frames are dropped (evidence integrity > detect completeness) and the drop counter increments.
- **Degradation ladder:** inference fails → camera ingestion + evidence buffer continue → camera AI status = `DEGRADED` → system-health warning. A camera never displays healthy-green when its AI pipeline is down.

## 6. Object detection

`services/inference` defines the only contract the rest of the system sees:

```
Detector (protocol)
├── info: ModelInfo {name, version, family, license, classes, input_size}
├── detect(frame) -> list[Detection]
├── warmup() / close()

DetectorRegistry.create(profile_name) -> Detector
```

- `Detection` uses **normalized bounding boxes** (0–1) so detection-stream and evidence-stream coordinates are interchangeable.
- Default class taxonomy: `person, vehicle, bag, weapon, ppe, other` — extensible per model profile; models may emit finer classes mapped through the profile.
- Adapters are **swappable**: `NullDetector` / `StubDetector` / `HogPeopleDetector` (`opencv-hog-people`, status `development`) now; ONNX / TensorRT / RF-DETR / RT-DETRv2 / YOLOX adapters later. **No model is downloaded in this phase** — the permissive-license shortlist and evaluation gate live in `DETECTOR_SELECTION.md`; no winner is selected.
- Existing Ultralytics weights are research assets only (see `MODEL_REGISTRY.md`); they do NOT become the default commercial detector, and their AGPL licensing must be resolved before any commercial deployment.

## 7. Multi-object tracking

`services/tracking` defines:

```
Tracker (protocol): update(detections, ts) -> list[TrackState]
```

`TrackState`: `track_id, class_name, confidence, bbox, first_seen, last_seen, trajectory[], age, hits, time_since_update, state(tentative|confirmed|lost)`.

- Implemented now: `IOUTracker` (**DEVELOPMENT TRACKER** — greedy IoU baseline, explicitly non-production, never a production profile) and `NullTracker`.
- Target: **ByteTrack** (default, cheap) and **BoT-SORT** (multi-camera/ReID needs), both behind the same interface — evaluation in the model-evaluation phase.
- Zone membership is NOT stored by the tracker; it is derived by the rule engine from track geometry + camera zones, keeping the tracker camera-agnostic.

## 8. Spatial intelligence

- `Zone`: per-camera named polygon (normalized coordinates), `zone_type` (restricted, hazardous, loading, entrance, emergency, parking, waiting, custom), enabled flag, metadata.
- Geometry ops (pure Python, `services/rules/geometry.py`): point-in-polygon, segment intersection (line crossing + direction), centroid.
- Rule conditions reference zones by ID; zones are drawn in the future Zone Editor UI and stored per camera.

## 9. Temporal intelligence

Complex activity is not a single-frame classification:

```
TemporalAnalyzer (protocol): analyze(clip_window, tracks) -> list[CandidateSignal]
ClipWindow {camera_id, start, end, frame_refs}
CandidateSignal {analyzer, activity, confidence, ts, window, metadata}
```

- Foundation ships only the interface + `NullAnalyzer`.
- Future adapters: VideoMAE-family clip classifiers, pose/temporal models (fall), lightweight VLM verification — all producing `CandidateSignal`, all versioned in the model registry.
- Analyzers emit **candidates**, never events: the event engine still applies confirmation, schedule, zone and cooldown logic.

## 10. Rule engine

- Rules consume an `EvaluationContext` (camera, time, schedule, zones, zone-enriched tracks, temporal state) and return an explainable `RuleMatch` or nothing.
- Reference rules implemented: `ZoneEnterRule`, `ZoneDwellRule`, `LineCrossRule`, plus `ScheduleCondition` (restricted-hours windows).
- Rule packs (`rules/packs/*.yaml`) configure industry behavior: `factory`, `warehouse`, `hospital` examples included. Same engine, different configuration.
- Every match carries `conditions: [{name, operator, actual, threshold, satisfied}]` → the exact "why" shown to operators (§11).

## 11. Event engine

```
RuleMatch → EventEngine.confirm()
  1. temporal confirmation   — must persist N seconds / M evaluations
  2. deduplication           — key = camera|type|zone|tracks; suppress while open
  3. cooldown                — per-rule quiet period after close
  4. severity (+ escalation) — from rule, optionally escalated on repeat/long dwell
  5. persist Event(NEW) → status machine
```

Event schema (implemented as `packages/schemas/event.py::Event`):

```
event_id, camera_id, timestamp, event_type, severity, status,
track_ids, zone_id, confidence, rule_id, conditions[] (explainability),
model_versions, evidence_ids[], metadata
```

States: `NEW → ACKNOWLEDGED → RESOLVED`, any → `DISMISSED` (audited).

**Worked examples (design intent):**

- Person + restricted zone + after-hours + dwell > 20 s + track confidence → `restricted_zone_intrusion` (critical), conditions list shows each term.
- Weapon classification + persistence across frames + high confidence + zone → `weapon_candidate` (never auto-escalated; human confirmation).
- Fall-like motion + body near ground + no recovery for N s → `fall_candidate`.

A 1-second zone clip during business hours must NOT become a critical event; that logic lives in confirmation + schedule thresholds, not in the detector.

## 12. Evidence capture (core feature)

Rolling in-memory buffer per camera (detect frames always; high-res evidence frames when available):

1. Event fires at T.
2. Emit snapshot(s) + clip `[T − pre_seconds … T + post_seconds]` (defaults 10 s / 10 s, configurable per camera/retention policy).
3. Attach metadata: timestamp, camera, event type, track IDs, zone, confidence, rule ID, model versions, event ID.
4. `sha256` hash + size + storage URI recorded in `Evidence`.
5. `expires_at` derived from the retention policy (severity → days), sweep deletes expired evidence.

`EvidenceStore` abstraction: `LocalDiskEvidenceStore` implemented; S3-compatible / enterprise object storage adapters deferred behind the same interface.

Ordinary frames are **never** written to disk.

## 13. Alert engine

Events ≠ notifications. `AlertRouter` maps an event to channels (webhook + `in_app` implemented in F2; dashboard/email/MQTT/SMS later) using per-rule/channel config, with:

- cooldown + dedup per `(camera, event_type, channel)` plus dispatch idempotency per `event_id`,
- acknowledgement,
- escalation: unacknowledged high/critical after N minutes → next channel,
- storm control: rate cap per camera with summary messages.

F2 delivery architecture: the pipeline submits each confirmed event to a bounded, non-blocking `AlertDispatcher` queue; a worker thread fans out to notifiers. Webhook delivery is fail-fast validated (http/https URL), bounded retries with exponential backoff, timeout-capped, and every failure message passes through `redact_secrets`. **No alert failure can crash or stall the pipeline** — drops/failures are counted and logged, events still persist. `in_app` alerts are rows in `alert` (queryable via `GET /api/v1/alerts`).

## 14. Event bus decision

| Option | Verdict | Why |
|---|---|---|
| **Redis Streams** | **chosen** | durable append log + consumer groups + replay; one infra component already useful as cache/rate-limit store; trivial ops; fits single-site edge box |
| NATS (JetStream) | later if needed | stronger multi-site fanout, but adds a second broker before we need it |
| MQTT | reserved for device integration | right for camera/IoT control plane, weak for replayable processing pipelines |
| Celery/RQ task queues | rejected for backbone | heavy dependency, poor fit for continuous event streams; may return for batch jobs |

Interface (`packages/common/bus.py`) is broker-agnostic: `InMemoryBus` (tests/dev default), `RedisBus` (compose/production). Frames never cross the bus (§3).

## 15. Database decision — PostgreSQL

PostgreSQL is the default relational store:

- JSONB for event conditions, trajectories, health details, rule definitions — flexible evidence metadata without schema churn.
- Strong durability + concurrency for many camera workers writing health rows while operators query events.
- Row-level locking for event state transitions; `LISTEN/NOTIFY` as a lightweight internal push channel.
- Mature ecosystem: Alembic migrations, pgvector (later: semantic search over event metadata), partitioning for event/evidence tables at scale.
- SQLite is used only as a **dev/test convenience default** so the foundation runs without Docker.

Entities: `Organization, Site, Camera, CameraHealth, Zone, Rule, RuleProfile, Event, Track, Evidence, Alert, User, Role, AuditLog, Model, ModelVersion, Deployment` — see `packages/db/models.py`.

Multi-tenancy: `Organization → Site → Camera` scoping on all operational rows; API filters every query by the principal's organization.

## 16. API

- FastAPI, mounted under `/api/v1` (versioning by path).
- Groups: `/auth /cameras /cameras/{id}/health /cameras/{id}/zones /zones /rules /events /events/{id} /events/{id}/status /evidence /evidence/{id}/download /alerts /models /system/health /search`.
- Pydantic schemas shared through `packages/schemas`; structured errors `{error: {code, message, details, request_id}}`; limit/offset pagination; filters on list endpoints.
- Implemented (F1 + F2): health, `/auth/me`, cameras (full CRUD + soft delete + health history), zones (full CRUD), rules (full CRUD), events (search/filtering + state transitions), evidence (list/detail + integrity-checked download), alerts (list). `/models` and `/search` remain `501` skeletons with the same error envelope. Reference: `API_REFERENCE.md`.

## 17. Security architecture

- **AuthN:** `api_key` mode (hashed comparison, `X-API-Key` header) in foundation; OIDC/password login deferred. `disabled` mode only for local dev, logs a warning at startup.
- **AuthZ:** RBAC — roles `viewer | operator | admin`, permission strings (`events:read`, `events:ack`, `cameras:manage`, `audit:read`…), per-camera permissions reserved in the data model and enforced in the API dependency layer as it is implemented.
- **Secrets:** RTSP URLs/credentials come from config/env, never logged; `stream_url` marked secret in logs.
- **Audit:** `AuditLog` for camera/zone/rule mutations, evidence download and retention outcomes (actor = API principal or `system:retention`); denied requests write no audit rows. Authentication-event auditing is reserved for the OIDC/login phase.
- **Hard rules:** no face recognition by default (separate high-risk module + explicit product/legal review if ever proposed); encrypted secret storage and secure exports are roadmap items with design hooks in place; rate limiting implemented as a per-principal sliding window.
- Reference lesson: production NVRs (e.g. Frigate 2025–26 CVEs) show that viewer/admin boundary bugs leak camera credentials — RBAC checks belong in one shared dependency, not scattered in handlers.

## 18. Observability

Metrics (design; API exposes health checks now): camera FPS, frame drops, reconnects, inference latency p50/p95/p99, queue depth, GPU/CPU/memory, event rate, alert rate, model version in use, error counts.

Health states: `HEALTHY | DEGRADED | OFFLINE | ERROR`.

`GET /api/v1/system/health` returns overall status + per-check results (`database`, `bus`, `process`) with HTTP 200 when healthy, 503 otherwise.

## 19. Deployment topologies

| Mode | Contents |
|---|---|
| Local dev | `deployments/docker/docker-compose.yml` (api + postgres + redis) or bare `uvicorn` on SQLite |
| Edge | single GPU/CPU box on-site, same images, camera workers + inference in one host, local disk evidence |
| On-prem | enterprise server/NVR, Postgres + object storage, SSO integration |
| Cloud-optional | central management/fleet only — **core monitoring works fully offline/on-prem; raw video is not sent to any cloud by default** |

## 20. Foundation scope boundary

**Implemented through F1:** structure, config layer, logging, schemas, DB models + migrations, FastAPI skeleton with auth/RBAC hooks/errors/pagination, health endpoint, camera abstraction + state machine + sources, detector interface + registry (HOG development detector), tracker interface + IOU baseline, temporal analyzer interface, rule engine + geometry + 3 reference rules + rule packs, event engine (confirm/dedup/cooldown/states), evidence buffer + store + retention logic, alert router + notifier interfaces, bus abstraction, Docker dev environment, tests.

**Implemented in F2 (see §21):** camera/zone/rule CRUD, event search, evidence retention operations + download, alert delivery (webhook + in_app + dispatcher), evaluation framework, detector research gate, hardened docs, authz/audit/observability coverage.

**Deliberately NOT implemented yet:** real model inference/downloads/training, model selection (research gate only — no winner), ONVIF, ByteTrack/BoT-SORT, temporal/VLM analyzers, email/MQTT notifier transports (interfaces only), S3 store (interface only), OIDC, semantic search, recording/NVR continuous storage, production deployment hardening. The React console exists (§22) but has no server-side rendering/SSO; it is a static bundle served by any web server or the Vite dev server.

Next phase recommendations: see `DEVELOPMENT_ROADMAP.md`.

## 21. F2 phase — what was built

F2 turned the read-only skeleton into an operator-configurable system, always behind RBAC and audit:

| Area | Delivered |
|---|---|
| **Cameras** | full CRUD; soft delete (`deleted_at`, disabled for detection); `stream_url` write-only (`stream_url_set`); `onvif`/bad-scheme sources rejected with `422`; re-create of deleted id → `409`; per-camera health history |
| **Zones** | per-camera CRUD; normalized simple polygons (self-intersection → `422 invalid_polygon`), unique name per camera, anchors (`center/top_center/bottom_center`); zone→camera resolves the business `camera_id` |
| **Rules** | CRUD for the four implemented rule types (`zone_enter`, `zone_dwell`, `line_cross`, `restricted_zone_intrusion`); immutable `rule_id`, `version` bumps only on change; site/camera validation; `400 unsupported_rule_type` for research types |
| **Event search** | filters `camera_id, status, severity, event_type, rule_id` + time range (`since/until` or `start_time/end_time`), deterministic `timestamp desc, id desc` pagination |
| **Evidence retention** | `RetentionService` + `scripts/run_retention_sweep.py` (`--dry-run`): file-before-row ordering, idempotent, `file_missing` handling, active-event protection (`allow_active_event_deletion`), audits `evidence.retention_delete`/`_failed` as `system:retention` |
| **Evidence download** | `GET /evidence/{id}/download` (`evidence:export`): local-backend only, path containment check, sha256 verification → `409 integrity_mismatch`, `X-Checksum-SHA256` + attachment headers, audited |
| **Alerts** | `AlertSettings` (`SENTINEL_ALERTS__*`); bounded non-blocking `AlertDispatcher`; `WebhookNotifier` (fail-fast URL, bounded retries + backoff, redacted errors); `DatabaseNotifier` (`in_app` rows); dispatch idempotency per `event_id`; `GET /alerts`; alert failures never touch the pipeline |
| **Evaluation** | `evaluation/` package: manifest + annotation schemas, IoU/precision/recall/F1/mAP metrics, runner with latency p50/p95/p99 and error examples; missing data → `EVALUATION DATASET NOT AVAILABLE` with **null metrics** (never fabricated); CLI runner exits 3 when unavailable |
| **Detector research gate** | `DETECTOR_SELECTION.md`: RT-DETRv2 / RF-DETR / YOLOX license + framework research, shortlist (no winner), evaluation plan; AGPL and PML-weight exclusions documented |
| **Model registry** | `MODEL_REGISTRY.md` hardened: closed status vocabulary `candidate|development|evaluated|approved|deprecated`, required `framework/task/intended-use` fields, HOG = `development` (never evaluated); enforced by `tests/test_f2_docs_contract.py` |
| **AuthZ / audit / logs** | permission matrix verified for viewer/operator/admin across all endpoints (tests), audit rows for every mutation with caller as actor, `unhandled_error_handler` logs traceback with secrets redacted |

## 22. F3 phase — operator experience console

F3 added the operator-facing web console (`apps/web/`) plus the minimal,
additive backend surface it needs. No new AI, no media server, no new
infrastructure.

**Backend additions (additive, all tested in `tests/test_f3_api.py`):**

| Piece | Purpose |
|---|---|
| `GET /stream` (SSE) | topics `events.created`, `events.updated`, `camera.health`, `alerts.updated`; `events:read`; recent-message replay; ~15 s heartbeat; async generator polling the in-memory bus off-thread (a blocking queue read would stall client-disconnect handling) |
| `GET /cameras/health/summary` | latest health row per camera with a derived `status` (`online/degraded/retrying/offline/disabled/unknown`) and `stale` flag (no health for 60 s); excludes soft-deleted cameras |
| `GET /cameras/{id}/preview.jpg` | buffered development frame from the in-process rolling buffer; `409 preview_unavailable` when no pipeline runs; never called "live video" |
| Bus publishes | `events.updated` after commit in `update_event_status`, `camera.health` on runner health flush, `alerts.updated` after dispatch — the UI reacts instead of guessing |

**Frontend architecture (`apps/web`, Vite + React 19 + TS + Tailwind 4):**

- `lib/api.ts` — single typed client; `X-API-Key` header; error envelopes →
  `ApiError` (status/code/request_id); `fetchSystemHealth` accepts `503` because
  degraded health is still a valid body.
- `lib/auth.tsx` — `/auth/me`-driven auth context; `api_key` mode stores the key
  in `localStorage` and shows a login gate; `disabled` mode needs no key.
  Permissions only *hide* controls; every endpoint is still enforced server-side.
- `lib/useLiveUpdates.ts` + `lib/live.tsx` — fetch-based SSE parser (native
  `EventSource` cannot send `X-API-Key`) with an automatic 10 s polling
  fallback; header badge is honest: `LIVE` (SSE), `POLLING · 10s`, `OFFLINE`.
- Pages: Command Center (dashboard), Camera Wall (status filters, add/edit,
  two-step delete), Camera Detail (preview + overview/zones), Event Feed
  (URL-param filters + pagination), Event Detail (ConditionEvidence table,
  status actions, evidence panel), Alert Center, Rule Configuration (four
  supported rule types, params/line/zone editing), Health (system checks +
  per-camera snapshots), SVG Zone Editor (draw in frame pixels, store
  normalized 0–1, ≥3 points client-side, server still enforces simplicity).
- **Honest labels:** camera tiles are `PREVIEW`/`UNAVAILABLE` (no media
  server, so never "live video"); the `LIVE` badge refers only to the event
  stream; explanations render stored `conditions` only — nothing is invented.
- Resilience: loading/empty/error/unauthorized states on every screen, a
  top-level `ErrorBoundary` so one broken panel cannot blank the console.

**Tests:** backend `tests/test_f3_api.py` (19) + bus unsubscribe (2);
frontend 38 vitest tests across 11 files (API client, auth gate, all pages,
zone validation, evidence gating). Full suite: 205 backend + 38 frontend.

## 23. F4 phase — real detector + evaluation gate + inference integration

F4 put a real, license-screened detector behind the existing `Detector`
contract and built the trustworthy-evaluation gate around it. No new AI
capabilities beyond detection, no accuracy claims, no dataset downloads.

| Area | Delivered |
|---|---|
| **Detector contract** | `Detection.class_id` added (provenance-complete, keyword-safe call sites); `DetectorError` moved to `interfaces.py` (shared contract, `hog` re-exports); contract documented in `DETECTOR_SELECTION.md` §7 (interface, wiring, failure isolation, replaceability rules) |
| **Candidate selection** | license/provenance gate (§8, 7 items + empirical inference contract), scored criteria table (§9) → **YOLOX-Tiny = first evaluation candidate** (provisional, engineering ordering, not accuracy) |
| **Evaluation data gate (F4-D)** | no qualifying licensed dataset → `EVALUATION DATASET NOT AVAILABLE`, null metrics, **no COCO download** (unauthorized substitution forbidden); `datasets/README.md` = acquisition spec |
| **Manifest contract (F4-E)** | `dataset_id`/`class_mapping`/`source_resolution`/`capture_context`, marked template, `validate_manifest.py` (8 checks, nothing silently discarded); 18 gate tests |
| **Adapter (F4-F)** | `services/inference/yolox.py`: YOLOX-Tiny ONNX behind the contract; empirical preproc (0–255 input — normalization baked into graph), external decode + **pixel-space NMS** (normalizing x/y by different frame dims distorts IoU and drops detections — found and fixed); `YOLOX_SPEC` pinned |
| **Weight management (F4-G)** | `scripts/weights.lock.json` (URL/tag/sha256/size), `download_models.py` (checksum-enforced, `--check`), `models/*.onnx` gitignored, `models/README.md` provenance; missing weights = deterministic config error, never a silent download |
| **Registry status (F4-Q)** | row 10 `yolox-tiny` = `candidate` (never `approved`); default profile stays `hog`; yolox opt-in only; enforced vocabulary via `tests/test_f2_docs_contract.py` |
| **Pipeline integration (F4-M/N)** | yolox profile through `default_registry()` → events carry `model_versions["detector"] = "yolox-tiny:0.1.1rc0"`; per-frame failure isolation proven (corrupt frames → `detector_errors`, `degraded` → `healthy` recovery, worker survives); startup weight failure = clear config error while API continues |
| **Benchmarks (F4-I)** | `benchmark_pipeline.py` extended (`--warmup`, `--json`, full env/model provenance); dev-clip reference run: p50 34.9 / p95 48.8 / p99 74.6 ms detector latency, 24.8 fps, 0 errors, CPU-only — latency only, never accuracy |
| **Evaluation mechanics (F4-J/K/L)** | `threshold_sweep.py` + CLI (rows + `best_f1_observation` but `operational_threshold: NOT EVALUATED`), error examples enriched with `iou` + `model_version` + expected/predicted class |
| **End-to-end verification (F4-O/P)** | `run_f4_event_verification.py` **28/28**: stock factory pack honestly does not confirm (1.3 s condition < 2 s window); `f4-verification.yaml` pack confirms events with yolox provenance, sha256-verified evidence, in-app alert, API retrieval; tracking record (32 gated frames, 8 tracks, 0 zone ID switches, in-zone conf mean 0.53) written to `var/benchmarks/f4-event-verification.json` |
| **Known limitation surfaced** | `EvidenceService` keeps one session per camera → of two near-simultaneous events only the later gets evidence (pre-existing, candidate F5 item) |

**Tests added:** 56 (`test_f4_evaluation_gate.py` 18, `test_f4_detector_adapter.py` 24,
`test_f4_benchmark_eval.py` 9, `test_f4_pipeline_integration.py` 5).

**Explicitly NOT done in F4:** accuracy evaluation (no licensed data),
threshold approval, `approved` status, violence/weapon/fall/PPE/VLM features,
face/ReID, ONVIF, cloud/K8s, auto-retraining, dataset downloads, AGPL model
migration, any "production-ready" claim for HOG or YOLOX-Tiny.
