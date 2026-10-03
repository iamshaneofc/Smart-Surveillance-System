# SENTINEL — API Reference (F2 + F3)

Base URL: `http://<host>:<port>/api/v1` (default `127.0.0.1:8000`; `SENTINEL_API_HOST`, `SENTINEL_API_PORT`).

Status: foundation phase — models and search endpoints intentionally return `501`.

## 1. Authentication

Send an API key in the header when `SENTINEL_AUTH_MODE=api_key`:

```
X-API-Key: <key>
```

Keys and roles come from `SENTINEL_AUTH_API_KEYS` (JSON list), e.g.:

```
SENTINEL_AUTH_MODE=api_key
SENTINEL_AUTH_API_KEYS='[{"key":"admin-key","user":"admin1","roles":["admin"]}]'
```

With `SENTINEL_AUTH_MODE=disabled` (default, development only) every request is
authenticated as an unrestricted principal and a warning is logged at startup.

### Roles → permissions

| Permission | viewer | operator | admin |
|---|---|---|---|
| `events:read` | yes | yes | yes |
| `events:ack` / `events:dismiss` | – | yes | yes |
| `cameras:read` | yes | yes | yes |
| `cameras:manage` | – | – | yes |
| `evidence:read` | yes | yes | yes |
| `evidence:export` (download) | – | – | yes |
| `zones:read` / `rules:read` / `alerts:read` | – | yes | yes |
| `zones:manage` / `rules:manage` | – | – | yes |
| `models:read`/`manage`, `users:manage`, `audit:read`, `system:manage` | – | – | yes |

Errors: `401 unauthorized` (missing/unknown key), `403 forbidden` (missing
permission), `429 rate_limited` (per-principal sliding window,
`SENTINEL_RATE_LIMIT_PER_MINUTE`, default 120).

## 2. Conventions

- **Pagination:** every list endpoint accepts `limit` (1–500, default 50) and
  `offset` (≥0) and returns
  `{"items": [...], "total": N, "limit": L, "offset": O}`.
- **Timestamps:** ISO-8601. Filtering accepts e.g.
  `2026-10-03T00:00:00Z` / `2026-10-03T12:00:00+00:00`.
- **Errors:**

  ```json
  {"error": {"code": "conflict", "message": "...", "details": {},
             "request_id": "..."}}
  ```

  Codes seen: `bad_request` (400), `unauthorized` (401), `forbidden` (403),
  `not_found` (404), `conflict` (409), `validation_error` (422),
  `rate_limited` (429), `internal_error` (500), `not_implemented` (501).
- **Stream URLs are never echoed:** responses expose `stream_url_set: true`;
  the raw URL is write-only (and redacted from logs/errors).

## 3. Health & identity

```bash
# Health (no permission required)
curl -s http://127.0.0.1:8000/api/v1/system/health

# Who am I
curl -s -H "X-API-Key: viewer-key" http://127.0.0.1:8000/api/v1/auth/me
```

`GET /auth/me` returns the principal with its effective `permissions`.

## 4. Cameras

```bash
# Create (source_type: rtsp | file | webcam | synthetic; onvif -> 422)
curl -s -X POST http://127.0.0.1:8000/api/v1/cameras \
  -H "X-API-Key: admin-key" -H "Content-Type: application/json" \
  -d '{"camera_id":"cam_gate","name":"Front gate","source_type":"rtsp",
       "stream_url":"rtsp://user:secret@10.0.0.10/stream1",
       "site_id":"site-a","metadata":{"floor":"1"}}'

# List (soft-deleted excluded by default)
curl -s "http://127.0.0.1:8000/api/v1/cameras?enabled=true&limit=50&offset=0" \
  -H "X-API-Key: viewer-key"
curl -s "http://127.0.0.1:8000/api/v1/cameras?include_deleted=true" \
  -H "X-API-Key: admin-key"

# Detail, update, delete (soft delete: deleted_at + disabled), health history
curl -s http://127.0.0.1:8000/api/v1/cameras/cam_gate -H "X-API-Key: viewer-key"
curl -s -X PATCH http://127.0.0.1:8000/api/v1/cameras/cam_gate \
  -H "X-API-Key: admin-key" -H "Content-Type: application/json" \
  -d '{"name":"Front gate 2","enabled":false}'
curl -s -X DELETE http://127.0.0.1:8000/api/v1/cameras/cam_gate -H "X-API-Key: admin-key"
curl -s "http://127.0.0.1:8000/api/v1/cameras/cam_gate/health?limit=20" \
  -H "X-API-Key: viewer-key"
```

Notes: `camera_id` must match `^[a-zA-Z0-9_\-]+$` (1–64). Re-creating a
soft-deleted id → `409 conflict`. `rtsp` requires `rtsp://`/`rtsps://`;
`file` requires a path; `onvif` → `422 validation_error`. Filters:
`enabled`, `site_id`, `include_deleted`. Audited: `camera.create`,
`camera.update`, `camera.delete`.

### Camera health summary & frame preview (F3)

```bash
# Latest health row per camera (excludes soft-deleted)
curl -s http://127.0.0.1:8000/api/v1/cameras/health/summary -H "X-API-Key: viewer-key"

# Development frame preview (JPEG from the in-process pipeline buffer)
curl -s -o frame.jpg -D - \
  -H "X-API-Key: viewer-key" \
  http://127.0.0.1:8000/api/v1/cameras/cam_gate/preview.jpg
```

Summary rows: `{camera_id, name, location, site_id, enabled, status, stale,
state, health, ai_status, fps, frame_drops, reconnect_count, last_frame_at,
error, health_ts}` where `status ∈ online|degraded|retrying|offline|disabled|
unknown` and `stale: true` when `health_ts` is missing or older than 60 s.

Preview responses: `200` with `X-Frame-Timestamp`, `Cache-Control: no-store`
and `X-Preview-Mode: preview`; `404` for unknown cameras; `409
preview_unavailable` when no pipeline is running for the camera. This is a
buffered development frame — not a media stream (permission: `cameras:read`).

## 5. Zones

```bash
# Create under a camera (polygon: normalized [x,y] points, >= 3, simple)
curl -s -X POST http://127.0.0.1:8000/api/v1/cameras/cam_gate/zones \
  -H "X-API-Key: admin-key" -H "Content-Type: application/json" \
  -d '{"name":"Restricted area","zone_type":"restricted",
       "polygon":[[0.1,0.1],[0.4,0.1],[0.4,0.5],[0.1,0.5]],
       "anchor":"center"}'

# List (global or per camera), detail
curl -s "http://127.0.0.1:8000/api/v1/zones?camera_id=cam_gate" -H "X-API-Key: operator-key"
curl -s http://127.0.0.1:8000/api/v1/zones/<zone_id> -H "X-API-Key: operator-key"

# Update / delete
curl -s -X PATCH http://127.0.0.1:8000/api/v1/zones/<zone_id> \
  -H "X-API-Key: admin-key" -H "Content-Type: application/json" \
  -d '{"name":"Restricted area v2"}'
curl -s -X DELETE http://127.0.0.1:8000/api/v1/zones/<zone_id> -H "X-API-Key: admin-key"
```

Notes: coordinates normalized 0–1; self-intersecting polygons →
`422 invalid_polygon`; duplicate name per camera → `409`;
`anchor` ∈ `center|top_center|bottom_center`. Audited: `zone.create`,
`zone.update`, `zone.delete`.

## 6. Rules

```bash
# Create (rule_type must be implemented: zone_enter | zone_dwell |
# line_cross | restricted_zone_intrusion)
curl -s -X POST http://127.0.0.1:8000/api/v1/rules \
  -H "X-API-Key: admin-key" -H "Content-Type: application/json" \
  -d '{"rule_id":"perimeter-enter","name":"Perimeter enter",
       "rule_type":"zone_enter","event_type":"zone_enter",
       "severity":"high","camera_id":"cam_gate","zone_ids":["<zone_id>"],
       "cooldown_seconds":30}'

# List / detail (filter: rule_type, camera_id)
curl -s "http://127.0.0.1:8000/api/v1/rules?camera_id=cam_gate" -H "X-API-Key: operator-key"
curl -s http://127.0.0.1:8000/api/v1/rules/perimeter-enter -H "X-API-Key: operator-key"

# Update (version bumps only on real change) / delete
curl -s -X PATCH http://127.0.0.1:8000/api/v1/rules/perimeter-enter \
  -H "X-API-Key: admin-key" -H "Content-Type: application/json" \
  -d '{"severity":"critical"}'
curl -s -X DELETE http://127.0.0.1:8000/api/v1/rules/perimeter-enter -H "X-API-Key: admin-key"
```

Notes: duplicate `rule_id` → `409`; renaming `rule_id` via PATCH →
`409 rule_id_immutable`; unsupported `rule_type` → `400
unsupported_rule_type`; unknown camera/site → `404`/`422`. Version starts at
`"1"`. Audited: `rule.create`, `rule.update`, `rule.delete`.

## 7. Events

```bash
# Search + pagination (filters: camera_id, status, severity, event_type,
# rule_id, time range via since/until or start_time/end_time)
curl -s "http://127.0.0.1:8000/api/v1/events?status=new&severity=high&limit=25&offset=0" \
  -H "X-API-Key: viewer-key"
curl -s "http://127.0.0.1:8000/api/v1/events?rule_id=perimeter-enter\
&start_time=2026-10-01T00:00:00Z&end_time=2026-10-04T00:00:00Z" \
  -H "X-API-Key: viewer-key"

# Detail
curl -s http://127.0.0.1:8000/api/v1/events/<event_id> -H "X-API-Key: viewer-key"

# Lifecycle: acknowledged/resolved need events:ack, dismissed needs events:dismiss
curl -s -X POST http://127.0.0.1:8000/api/v1/events/<event_id>/status \
  -H "X-API-Key: operator-key" -H "Content-Type: application/json" \
  -d '{"status":"acknowledged"}'
```

Ordering is deterministic (`timestamp desc, id desc`). Invalid lifecycle
transitions → `409 invalid_transition`. Events carry explainability
(`conditions`, `summary`), `model_versions`, `rule_id`/`rule_version`, and
`evidence_ids`.

## 8. Evidence

```bash
# List (filter: event_id, camera_id) ordered by captured_at desc
curl -s "http://127.0.0.1:8000/api/v1/evidence?event_id=<event_id>" \
  -H "X-API-Key: viewer-key"

# Detail (includes sha256, size_bytes, expires_at)
curl -s http://127.0.0.1:8000/api/v1/evidence/<evidence_id> -H "X-API-Key: viewer-key"

# Download (evidence:export = admin only)
curl -s -D - -o clip.mp4 \
  -H "X-API-Key: admin-key" \
  http://127.0.0.1:8000/api/v1/evidence/<evidence_id>/download
```

Download responses carry `X-Checksum-SHA256` and
`Content-Disposition: attachment`. Integrity mismatch (file vs stored hash)
→ `409 integrity_mismatch`; missing file → `404`; non-local storage backend →
`409 unsupported_backend`. Every successful download is audited as
`evidence.download`.

## 9. Alerts

```bash
# Persisted alert deliveries (filter: event_id, channel, status)
curl -s "http://127.0.0.1:8000/api/v1/alerts?status=sent&limit=50" \
  -H "X-API-Key: operator-key"
```

Alert delivery itself is pipeline-internal (see `docs/OPERATIONS.md` §3);
failures never fail event processing.

## 10. Server-sent events (F3)

```bash
# Event stream (events:read). Requires the X-API-Key header, hence a
# fetch-based SSE client instead of native EventSource.
curl -N -H "X-API-Key: viewer-key" -H "Accept: text/event-stream" \
  http://127.0.0.1:8000/api/v1/stream

# Optional topic filter (repeatable): events.created, events.updated,
# camera.health, alerts.updated
curl -N -H "X-API-Key: viewer-key" "http://127.0.0.1:8000/api/v1/stream?topics=camera.health"
```

Wire format: `: connected` comment on open, `id:`/`event:`/`data:` frames per
topic, a `: heartbeat` comment every ~15 s, and up to 5 recent messages per
topic replayed on connect. Publishers: event create/update (after commit),
camera health flushes, alert dispatch results. The UI falls back to 10 s
polling when the stream cannot stay open.

## 11. Not implemented yet (501)

```bash
curl -s http://127.0.0.1:8000/api/v1/models -H "X-API-Key: admin-key"   # 501
curl -s "http://127.0.0.1:8000/api/v1/search?q=person" \
  -H "X-API-Key: viewer-key"                                            # 501
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login                 # 501
```

## 12. Not in the API (by design)

No VLM/natural-language search, no face/ReID, no ONVIF control, no
NL/semantic event search, no model download/activation endpoints. Detector
selection is a documented research gate: `docs/DETECTOR_SELECTION.md`.
There is no media-server/HLS/WebRTC endpoint: video exposure is the
development frame preview only (§4).
