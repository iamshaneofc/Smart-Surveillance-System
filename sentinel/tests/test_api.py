from datetime import datetime, timezone

from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus


def _seed_event(session, event_id="evt_api1", camera_id="cam1", severity=Severity.HIGH):
    from services.events.repository import SqlEventRepository

    now = datetime(2026, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
    event = Event(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=now,
        event_type="restricted_zone_intrusion",
        severity=severity,
        status=EventStatus.NEW,
        confidence=0.91,
        track_ids=[182],
        zone_id="z1",
        zone_name="Restricted",
        rule_id="restricted-zone-entry",
        rule_name="Restricted zone entry",
        conditions=[
            {"name": "zone", "operator": "in", "actual": "Restricted", "threshold": "z1", "satisfied": True}
        ],
        model_versions={"detector": "stub:1"},
        created_at=now,
        updated_at=now,
    )
    SqlEventRepository(session).save(event)
    return event


def _seed_camera(session, camera_id="cam1"):
    from packages.db import models

    row = models.Camera(
        camera_id=camera_id,
        name="Gate camera",
        location="North gate",
        source_type="rtsp",
        stream_url="rtsp://admin:secret@10.0.0.10/stream1",
        retention={},
    )
    session.add(row)
    session.flush()
    return row


def test_root_reports_foundation_phase(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "SENTINEL"
    assert body["phase"] == "foundation"


def test_health_healthy(client):
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["checks"]["database"]["status"] == "healthy"
    assert body["checks"]["event_bus"]["status"] == "healthy"
    assert body["app"]["version"]


def test_request_id_header_echoed(client):
    response = client.get("/api/v1/system/health", headers={"X-Request-ID": "req-test-1"})
    assert response.headers["X-Request-ID"] == "req-test-1"


def test_whoami_disabled_mode(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 200
    body = response.json()
    assert body["user"] == "dev-admin"
    assert body["auth_mode"] == "disabled"
    assert "events:read" in body["permissions"]
    assert body["warnings"]


def test_api_key_mode_requires_key(api_key_client):
    response = api_key_client.get("/api/v1/auth/me")
    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "unauthorized"
    assert error["request_id"]


def test_api_key_mode_valid_key_roles(api_key_client):
    response = api_key_client.get(
        "/api/v1/auth/me", headers={"X-API-Key": "operator-key"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user"] == "operator1"
    assert "events:ack" in body["permissions"]


def test_api_key_viewer_cannot_ack(api_key_client):
    response = api_key_client.post(
        "/api/v1/events/does-not-exist/status",
        headers={"X-API-Key": "viewer-key"},
        json={"status": "acknowledged"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_rate_limit(settings):
    from fastapi.testclient import TestClient

    from apps.api.main import create_app
    from packages.db import base as db_base

    settings.rate_limit_per_minute = 2
    db_base.dispose()
    db_base.configure(settings.database_url)
    db_base.create_all()
    application = create_app(settings)
    with TestClient(application) as limited:
        assert limited.get("/api/v1/auth/me").status_code == 200
        assert limited.get("/api/v1/auth/me").status_code == 200
        blocked = limited.get("/api/v1/auth/me")
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "rate_limited"
        assert limited.get("/api/v1/system/health").status_code == 200
    application.state.bus.close()
    db_base.dispose()


def test_events_list_get_and_filter(client):
    from packages.db import base as db_base

    with db_base.session_scope() as session:
        _seed_event(session, "evt_a", "cam1", Severity.HIGH)
        _seed_event(session, "evt_b", "cam2", Severity.LOW)

    response = client.get("/api/v1/events")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2

    filtered = client.get("/api/v1/events", params={"camera_id": "cam1"})
    assert filtered.json()["total"] == 1

    by_severity = client.get("/api/v1/events", params={"severity": "low"})
    assert by_severity.json()["items"][0]["event_id"] == "evt_b"

    detail = client.get("/api/v1/events/evt_a")
    assert detail.status_code == 200
    assert detail.json()["status"] == "new"
    assert detail.json()["conditions"]
    assert detail.json()["model_versions"] == {"detector": "stub:1"}


def test_event_not_found_envelope(client):
    response = client.get("/api/v1/events/nope")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert "nope" in error["message"]


def test_event_status_transition_and_audit(client):
    from packages.db import base as db_base, models
    from sqlalchemy import func, select

    with db_base.session_scope() as session:
        _seed_event(session, "evt_t1", "cam1")

    ack = client.post(
        "/api/v1/events/evt_t1/status", json={"status": "acknowledged", "note": "checking"}
    )
    assert ack.status_code == 200
    body = ack.json()
    assert body["status"] == "acknowledged"
    assert body["acknowledged_by"] == "dev-admin"

    invalid = client.post("/api/v1/events/evt_t1/status", json={"status": "acknowledged"})
    assert invalid.status_code == 409
    assert invalid.json()["error"]["code"] == "invalid_transition"

    with db_base.session_scope() as session:
        audits = session.execute(
            select(func.count()).select_from(models.AuditLog).where(
                models.AuditLog.action == "event.acknowledged"
            )
        ).scalar_one()
    assert audits == 1


def test_event_validation_error(client):
    response = client.get("/api/v1/events", params={"limit": 0})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_cameras_endpoints_do_not_leak_stream_url(client):
    from packages.db import base as db_base

    with db_base.session_scope() as session:
        _seed_camera(session, "cam1")

    listing = client.get("/api/v1/cameras")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["stream_url_set"] is True
    assert "stream_url" not in item
    assert "secret" not in str(item)

    detail = client.get("/api/v1/cameras/cam1")
    assert detail.status_code == 200
    assert detail.json()["camera_id"] == "cam1"

    missing = client.get("/api/v1/cameras/nope")
    assert missing.status_code == 404


def test_camera_health_history(client):
    from packages.db import base as db_base, models

    with db_base.session_scope() as session:
        _seed_camera(session, "cam1")
        session.add(
            models.CameraHealth(
                camera_id="cam1",
                state="streaming",
                health="healthy",
                fps=4.8,
                ai_status="degraded",
            )
        )

    response = client.get("/api/v1/cameras/cam1/health")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["fps"] == 4.8
    assert items[0]["ai_status"] == "degraded"


def test_evidence_endpoints(client):
    from packages.db import base as db_base, models

    now = datetime(2026, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
    with db_base.session_scope() as session:
        _seed_event(session, "evt_e1", "cam1")
        session.add(
            models.Evidence(
                event_id="evt_e1",
                camera_id="cam1",
                type="clip",
                uri="cam1/evt_e1/clip.mp4",
                sha256="a" * 64,
                size_bytes=123,
                content_type="video/mp4",
                captured_at=now,
            )
        )

    response = client.get("/api/v1/evidence", params={"event_id": "evt_e1"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["type"] == "clip"
    assert body["items"][0]["sha256"] == "a" * 64

    download = client.get("/api/v1/evidence/whatever/download")
    assert download.status_code == 404
    assert download.json()["error"]["code"] == "not_found"


def test_stub_endpoints_return_501(client):
    for path in ("/api/v1/models", "/api/v1/search"):
        response = client.get(path)
        assert response.status_code == 501, path
        assert response.json()["error"]["code"] == "not_implemented"


def test_camera_create_requires_supported_source(client):
    response = client.post("/api/v1/cameras", json={"camera_id": "x", "name": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"

    onvif = client.post(
        "/api/v1/cameras",
        json={"camera_id": "x", "name": "x", "source_type": "onvif", "stream_url": "rtsp://cam"},
    )
    assert onvif.status_code == 422
    assert onvif.json()["error"]["code"] == "validation_error"
