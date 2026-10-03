# SENTINEL — Development Roadmap

**Completed phases are marked COMPLETE with their exit evidence. Phases after F2 must be reviewed and approved before starting. This document is a plan, not a claim of completion for unfinished phases.**

## F0 — Foundation (COMPLETE)

- [x] Monorepo structure (`apps/ services/ packages/ …`)
- [x] Config layer (env-driven, typed)
- [x] Structured logging + request IDs
- [x] Shared Pydantic schemas (camera, event, evidence, rule, auth, track)
- [x] PostgreSQL data model (15+ entities) + Alembic initial migration
- [x] FastAPI skeleton: `/api/v1`, RBAC dependencies, structured errors, pagination, rate limiting
- [x] Health endpoint with per-check degradation (`HEALTHY/DEGRADED/OFFLINE/ERROR`)
- [x] Camera abstraction: sources, state machine, health stats, reconnect/backoff
- [x] Detector interface + registry (no model downloads)
- [x] Tracker interface + IOU baseline (no ByteTrack yet)
- [x] Temporal analyzer interface (no analyzers yet)
- [x] Rule engine: geometry, schedules, zone-enter/dwell/line-cross reference rules, explainable conditions
- [x] Rule packs: factory / warehouse / hospital YAML examples
- [x] Event engine: confirmation, dedup, cooldown, severity, state machine
- [x] Evidence abstraction: rolling pre/post buffer, clip/snapshot assembly, hash, retention logic, local-disk store
- [x] Alert router + notifier interfaces (no transports yet)
- [x] Bus abstraction (in-memory + Redis Streams)
- [x] Docker dev environment (api + postgres + redis)
- [x] Tests for engine/rules/evidence/camera/API
- [x] Docs: architecture, model registry, data licenses, asset disposition, this roadmap

**Exit criteria met:** code runs, tests pass, no model/accuracy claims, legacy project untouched.

## F1 — Camera ingestion vertical slice (COMPLETE)

Deliver: real RTSP/file/webcam ingestion running as a supervised worker, `CameraHealth` persisted, detect-stream sampling with bounded queues, graceful shutdown, live health in API.

- Acceptance: stream survives router blips (reconnect ≤ backoff cap), health endpoint shows real FPS/drops, killing inference does not stop ingestion (isolation demo), one file-source e2e test on CI.

**Exit evidence:** full test suite green; `scripts/run_live_acceptance.py` **17/17** on a real video (HOG detector, one explainable event, hashed evidence); F1 report delivered.

## F2 — Operator configuration, evidence operations, alerts & evaluation (COMPLETE — this phase)

Deliver (per approved F2 spec, sections A–O):

- [x] **A** Camera CRUD (soft delete, `stream_url` write-only, source validation, audits, authz)
- [x] **B** Zone CRUD (simple polygons, anchors, per-camera unique names, audits)
- [x] **C** Rule CRUD (implemented types only, immutable `rule_id`, version bumping, audits)
- [x] **D** Event search (filters incl. `rule_id`, time ranges, deterministic pagination)
- [x] **E** Evidence retention service + `scripts/run_retention_sweep.py` (idempotent, auditable, active-event protection)
- [x] **F** Evidence download (integrity-checked, path containment, audited, `evidence:export`)
- [x] **G** Alert architecture (bounded dispatcher, webhook retries/backoff/redaction, `in_app` persistence, `GET /alerts`, pipeline never fails on alerts)
- [x] **H** Evaluation framework (`evaluation/`: manifest, annotations, metrics, runner, CLI; unavailable data → null metrics)
- [x] **I** Evaluation docs + template manifest
- [x] **J** Detector research gate → `DETECTOR_SELECTION.md` (RT-DETRv2 / RF-DETR / YOLOX shortlist, **no winner**, evaluation plan)
- [x] **K** Model registry hardened (`MODEL_REGISTRY.md`: closed status vocabulary, framework/task/intended-use, HOG = `development`) + doc-contract tests
- [x] **L** Authz verification (viewer/operator/admin matrix across all endpoints + audit-row assertions)
- [x] **M** Observability (unhandled-error logging with redaction, retention/alert/evaluation logs)
- [x] **N** Full regression: pytest green, `alembic check` clean, live acceptance **17/17**, legacy untouched
- [x] **O** Docs: `API_REFERENCE.md`, `OPERATIONS.md`, F2 architecture section, this roadmap

**Exit evidence:** 184 tests green; alembic clean; live acceptance 17/17; no model/accuracy claims anywhere; legacy project untouched. **F3 starts only after review/approval.**

## F3 — Operator experience phase (COMPLETE — executed per approved F3 spec)

Deliver (per approved F3 spec, sections A–T): a React/TS operator console plus
only the additive backend surface it needs (SSE stream, camera health summary,
frame preview, three bus publishes). Explicitly out of scope for this phase:
VLM/NL search, face/ReID, new detection types, cloud/K8s, MLflow, ONVIF
discovery, SMS/WhatsApp, mobile app, media-server infra, any new AI.

- [x] **A** Backend groundwork (bus unsubscribe, buffer latest frame, SSE `/stream`, `/cameras/health/summary`, `/cameras/{id}/preview.jpg`, publishes for `events.updated` / `camera.health` / `alerts.updated`) + `tests/test_f3_api.py`
- [x] **B** Frontend scaffold (`apps/web`: Vite 6 + React 19 + TS + Tailwind 4 + vitest), typed API client, auth abstraction (api_key gate / disabled mode), app shell
- [x] **C** Command Center dashboard (system status, camera health, recent events)
- [x] **D** Camera wall (status filters, previews, add/edit/delete with permission gating)
- [x] **E** Camera detail (frame preview, overview, zones tab, health sidebar)
- [x] **F** Event feed (URL-param filters, deterministic pagination)
- [x] **G** Event detail with explainability (stored `ConditionEvidence` table, rule/model versions, status actions)
- [x] **H** Evidence viewer (metadata always; preview/download only with `evidence:export`, matching the server 403)
- [x] **I** Alert center (channel/status/event filters, attempts, errors, pagination)
- [x] **J** Camera management UI (create/edit/two-step delete)
- [x] **K** SVG zone editor (draw in frame pixels → store normalized 0–1, ≥3-point client validation, server enforces simplicity)
- [x] **L** Rule management UI (four implemented types, params JSON, normalized line points, zone ids)
- [x] **M** Health detail page (system checks + per-camera snapshots)
- [x] **N** Real-time hook (fetch-based SSE with 10 s polling fallback; honest LIVE/POLLING/OFFLINE badge)
- [x] **O** Responsive layout (desktop + mobile nav, all screens)
- [x] **P** Error/degraded/offline handling everywhere (loading/empty/error/unauthorized states + error boundary)
- [x] **Q** Frontend tests (38 tests / 11 files)
- [x] **R** Full regression: backend 205 tests, `alembic check` clean, live acceptance **17/17**, frontend tests + `tsc && vite build` green, legacy repo untouched (0 tracked modifications)
- [x] **S** Docs (`API_REFERENCE.md` F3 endpoints, `OPERATIONS.md` §7 console guide, architecture §22, this roadmap, `apps/web/README.md`, root `README.md`)
- [x] **T** Product polish pass (navigation, naming, error messages, empty states, visual consistency, event-detail/evidence/health readability, permissions, responsive behavior — fixes applied, see final report)

**Exit evidence:** backend 205 tests green; frontend 40 tests + clean build; alembic clean; live acceptance 17/17; no model/accuracy claims anywhere; legacy project untouched (0 tracked files modified). Final 19-field report delivered in chat. **F4 starts only after review/approval.**

### Former F3 detector/tracker scope — NOT done (needs re-approval)

The pre-F3 plan for a detector vertical slice was superseded by the approved
operator-experience scope above; these items remain unstarted:

- [ ] First real `Detector` adapter selected **through** the evaluation gate (`DETECTOR_SELECTION.md` plan; license registry updated first; no AGPL path without sign-off)
- [ ] `IOU → ByteTrack` upgrade behind the `Tracker` interface
- [ ] GPU/CPU device selection + per-camera confidence thresholds
- [ ] Rule packs loadable into DB (beyond per-rule CRUD)
- [ ] After-hours/schedule negative acceptance (1-second business-hours entry → no event; after-hours walk-in → one explainable event with `[−pre … +post]` clip)
- [ ] Golden-clip tracking acceptance: tracked IDs stable across occlusion; model profile recorded on every event

## F4 — API completion + remaining UI screens (NEXT — needs review/re-scope)

F3 already delivered the console foundation (login gate, Command Center, camera wall/detail, event feed/detail, alerts, rules, health, evidence viewer). Remaining scope to re-approve before starting:

Deliver: remaining endpoints (`/models`, `/search` stubs → real), server-side auth login flow (`/auth/login` currently 501), per-camera permissions, audit log query endpoint; UI for whatever new endpoints land.

- Acceptance: operator can configure a camera + zone + rule entirely through UI and receive/acknowledge events (already demonstrated in F3 — re-verify after any re-scope).

## F5 — Golden set + model selection execution

Deliver: golden set (migrated legacy FPs + normal/suspicious/edge/low-light/occlusion/crowd/angle cases), run the F2 evaluation harness per `DETECTOR_SELECTION.md` §4 (video-level split, FP/hour, ECE, latency on target hardware), per-model registry promotion workflow.

- Acceptance: first detector promoted `candidate → evaluated` with recorded numbers; legacy 91.22% quarantined in docs only; regression suite gates model changes.

## F6 — Remaining alert transports + temporal/VLM layer

Deliver: email/MQTT notifiers with escalation tests (webhook + `in_app` shipped in F2), temporal analyzer (clip classifier) behind `TemporalAnalyzer`, optional second-stage VLM verification producing grounded explanations only, semantic search over structured events first (embeddings only where useful).

- Acceptance: alert storm test (cooldown/escalation), VLM output constrained to evidence-derived JSON with abstain option.

## F7 — Commercial hardening

Deliver: OIDC/SSO, encrypted secret store, per-camera RBAC enforcement, ONVIF discovery **only if actually tested**, S3 evidence store, performance/soak tests at target camera counts, threat model + external pen test, license sign-off package (`MODEL_REGISTRY.md` + `DATA_LICENSES.md` → approved), packaging for edge/on-prem.

- Acceptance checklist maps to product spec sections 26–28: connect existing cameras → configure rules → start monitoring, with tested compatibility claims only.

## Cross-cutting backlog (any phase)

- Git hygiene: LFS for binaries, purge or quarantine legacy 10–40 MB blobs (explicitly approved change only).
- Kubernetes/edge packaging beyond compose.
- pgvector-based semantic retrieval (after F3 re-scope stabilizes the schema).
- Continuous recording NVR mode (explicitly out of scope until product decides; current design is event-driven evidence only).
