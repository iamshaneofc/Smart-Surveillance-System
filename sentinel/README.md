# SENTINEL

**AI-Powered Video Surveillance & Security Intelligence Platform — F3 (OPERATOR EXPERIENCE) PHASE.**

Core principle: **continuous monitoring + event-driven evidence.** Ordinary frames are processed transiently and discarded; evidence is captured only when a configured rule fires.

> **This is not production-ready.** Phases F0–F3 deliver the foundation,
> ingestion slice, operator configuration APIs, and the operator web console:
> no approved production detector, no accuracy claims, no media server.
> See `docs/DEVELOPMENT_ROADMAP.md`.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/SENTINEL_ARCHITECTURE.md`](docs/SENTINEL_ARCHITECTURE.md) | Full architecture, layer responsibilities, decisions |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | HTTP API + SSE contract, auth, permissions |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Running API + web console, retention, alerts, logging |
| [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md) | Model inventory, license status, dispositions |
| [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) | Dataset provenance and commercial-use rules |
| [`docs/DEVELOPMENT_ROADMAP.md`](docs/DEVELOPMENT_ROADMAP.md) | Phases F0–F7 with acceptance criteria |
| [`docs/RESEARCH_ASSET_DISPOSITION.md`](docs/RESEARCH_ASSET_DISPOSITION.md) | What happens (and does not happen) to the legacy repo |

## Structure

```
sentinel/
├── apps/
│   ├── api/            FastAPI service (routers, auth, errors, health, SSE)
│   └── web/            React operator console (Vite + TS + Tailwind, vitest)
├── services/
│   ├── camera/         ingestion, sources, camera worker state machine
│   ├── inference/      Detector interface + registry (no models yet)
│   ├── tracking/       Tracker interface + IOU baseline
│   ├── temporal/       TemporalAnalyzer interface (future clip models/VLM)
│   ├── rules/          geometry, schedules, reference rules, pack loader
│   ├── events/         event engine: confirm/dedup/cooldown/states
│   ├── evidence/       rolling buffer, clip/snapshot assembly, retention
│   └── alerts/         alert router + notifier interfaces
├── packages/
│   ├── schemas/        shared Pydantic schemas (API contract)
│   ├── config/         typed settings (env-driven)
│   ├── common/         logging, ids, time, bus abstraction
│   └── db/             SQLAlchemy models + Alembic migrations
├── rules/packs/        industry rule profiles (YAML, config over code)
├── models/             weights land here (registry: docs/MODEL_REGISTRY.md)
├── evaluation/         evaluation framework + golden set (deferred)
├── datasets/           licensed datasets only (docs/DATA_LICENSES.md)
├── deployments/        docker-compose, Dockerfile, edge (placeholder)
├── docs/               architecture documentation
└── tests/              pytest suite
```

Import root is `sentinel/` itself: `from packages.schemas import Event`, `from services.rules import …`.

## Quickstart (local, no Docker)

```bash
cd sentinel
python3.12 -m venv .venv            # or any Python >= 3.11
.venv/bin/pip install -e ".[dev]"   # Windows: .venv\Scripts\pip
cp .env.example .env                # optional; SQLite defaults work out of the box
.venv/bin/uvicorn apps.api.main:app --reload
# open http://127.0.0.1:8000/api/v1/system/health
```

Run tests:

```bash
.venv/bin/pytest
```

## Web console (operator UI)

```bash
cd apps/web
npm install
npm run dev            # http://127.0.0.1:5173 (proxies /api -> :8000)
npm test               # vitest suite
npm run build          # tsc -b && vite build
```

Pages: Command Center, Camera Wall, Camera Detail, Event Feed/Detail,
Alert Center, Rule Configuration, Health. Roles (viewer/operator/admin)
hide controls; the backend still enforces every permission. Camera tiles
show `PREVIEW`/`UNAVAILABLE` (no media server — never "live video");
the header `LIVE` badge refers to the SSE event stream. Details and
limitations: `docs/OPERATIONS.md` §7.

## Quickstart (Docker Compose: api + postgres + redis)

```bash
cd deployments/docker
docker compose up --build
```

## Status of the legacy project

The parent repository (`Z:\Security_Surveillance_System`) still contains the original research experiments (mask, violence, weapon, action). They are **preserved untouched** as a research archive; none of their code is imported here. Disposition of every legacy asset: `docs/RESEARCH_ASSET_DISPOSITION.md`.
