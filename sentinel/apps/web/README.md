# SENTINEL Web UI (phase F3)

Operator console: Vite + React 19 + TypeScript + Tailwind CSS v4 + react-router 7.

## Commands

```bash
npm install       # install dependencies
npm run dev       # dev server (proxies /api -> http://localhost:8000)
npm test          # vitest + testing-library
npm run build     # tsc -b && vite build
npm run typecheck # tsc -b
```

`VITE_API_BASE` overrides the API base path (default `/api/v1`, served through
the dev proxy). `VITE_API_PROXY_TARGET` overrides the backend target.

## Design rules (phase F3)

- **Backend is the authority.** The UI hides controls the signed-in principal
  lacks, but every endpoint is still enforced server-side (401/403/429).
- **Honest video states.** The console has no media server: a camera tile is
  either `PREVIEW` (development frame from `/cameras/{id}/preview.jpg`) or
  `UNAVAILABLE`. Never labelled live. The header `LIVE` badge refers only to
  the SSE event stream.
- **Real-time** arrives over fetch-based SSE (`/api/v1/stream`, because native
  `EventSource` cannot send `X-API-Key`) with an automatic 10 s polling
  fallback; the badge shows `LIVE` / `POLLING · 10s` / `OFFLINE`.
- **Explainability** comes only from stored `Event.conditions`
  (`ConditionEvidence`) — the UI never invents scores or model outputs.

## Structure

```
src/
  components/   shell, states, login gate, preview, zone editor, evidence panel, error boundary
  pages/        one file per route (dashboard, cameras, events, alerts, rules, health)
  lib/          typed API client, auth context, live-updates hook, formatting
  test/         vitest setup + shared test utils (renderPage, mockFetch, role fixtures)
```

Tests: `npm test` runs 40 vitest tests across 11 files (API client, auth
gate, every page, zone validation/anchor/toggle, evidence download gating).

See `docs/API_REFERENCE.md` for endpoint contracts and
`docs/SENTINEL_ARCHITECTURE.md` §22 (F3 console architecture).
