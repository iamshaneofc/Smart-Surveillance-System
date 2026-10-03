import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from packages.common.timeutil import utcnow
from packages.db.base import get_session
from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus


def _seed_camera(session, camera_id="cam1", name="Gate camera", enabled=True, deleted=False):
    from packages.db import models

    row = models.Camera(
        camera_id=camera_id,
        name=name,
        location="North gate",
        source_type="rtsp",
        stream_url="rtsp://cam/stream",
        enabled=enabled,
        deleted_at=utcnow() if deleted else None,
        retention={},
    )
    session.add(row)
    session.flush()
    return row


def _seed_health(session, camera_id="cam1", state="streaming", health="healthy",
                 ai_status="healthy", ts=None, error=None):
    from packages.db import models

    row = models.CameraHealth(
        camera_id=camera_id,
        state=state,
        health=health,
        ai_status=ai_status,
        fps=5.0,
        error=error,
        ts=ts or utcnow(),
    )
    session.add(row)
    session.flush()
    return row


def _seed_event(session, event_id="evt_f3", camera_id="cam1"):
    from services.events.repository import SqlEventRepository

    now = datetime(2026, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
    event = Event(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=now,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        status=EventStatus.NEW,
        confidence=0.91,
        track_ids=[182],
        zone_id="z1",
        zone_name="Restricted",
        rule_id="restricted-zone-entry",
        rule_name="Restricted zone entry",
        conditions=[
            {"name": "zone", "operator": "in", "actual": "Restricted",
             "threshold": "z1", "satisfied": True}
        ],
        model_versions={"detector": "stub:1"},
        created_at=now,
        updated_at=now,
    )
    SqlEventRepository(session).save(event)
    return event


# -- SSE stream -----------------------------------------------------------
#
# Live SSE responses never end on their own, and TestClient cannot close one
# cleanly. These tests drive the ASGI app directly in a thread with a
# controllable receive/send channel so disconnect (and unsubscribe) is
# deterministic.


def _drive_stream(app, path="/api/v1/stream", query="", headers=None):
    import asyncio
    import threading

    outbox: list[dict] = []
    import queue as _queue

    q: _queue.Queue = _queue.Queue()

    async def receive():
        while True:
            try:
                return q.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.01)

    async def send(message):
        outbox.append(message)

    header_items = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": header_items,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "state": {},
    }

    def run():
        asyncio.run(app(scope, receive, send))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def disconnect():
        q.put({"type": "http.disconnect"})

    return thread, outbox, disconnect


def _wait(predicate, timeout=5.0, message="condition not met"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    raise AssertionError(message)


def _body_text(outbox) -> str:
    return b"".join(
        m.get("body", b"") for m in outbox if m["type"] == "http.response.body"
    ).decode()


def _start_response(outbox) -> dict:
    _wait(lambda: outbox and outbox[0]["type"] == "http.response.start",
          message="no response start")
    return outbox[0]


def test_stream_401_without_key_in_api_key_mode(api_key_client):
    response = api_key_client.get("/api/v1/stream")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_stream_viewer_allowed(api_key_client):
    thread, outbox, disconnect = _drive_stream(
        api_key_client.app, headers={"X-API-Key": "viewer-key"}
    )
    try:
        start = _start_response(outbox)
        assert start["status"] == 200
        assert "text/event-stream" in dict(start["headers"])[b"content-type"].decode()
        _wait(lambda: ": connected" in _body_text(outbox),
              message="stream never started")
    finally:
        disconnect()
        thread.join(timeout=5)
    assert not thread.is_alive(), "stream did not shut down on disconnect"


def test_stream_connect_then_delivers_published_message(app):
    bus = app.state.bus
    thread, outbox, disconnect = _drive_stream(app)
    try:
        _wait(lambda: ": connected" in _body_text(outbox),
              message="connected comment missing")
        bus.publish("events.created", {"event_id": "evt_stream1", "camera_id": "cam1"})
        _wait(lambda: "evt_stream1" in _body_text(outbox),
              message="published message never arrived on stream")
        text = _body_text(outbox)
        line = next(ln for ln in text.splitlines() if "evt_stream1" in ln)
        envelope = json.loads(line[len("data: "):])
        assert envelope["topic"] == "events.created"
        assert envelope["payload"]["event_id"] == "evt_stream1"
    finally:
        disconnect()
        thread.join(timeout=5)
    assert not thread.is_alive(), "stream did not shut down on disconnect"


def test_stream_replays_recent_history(app):
    app.state.bus.publish("events.created", {"event_id": "evt_replay1"})
    thread, outbox, disconnect = _drive_stream(app)
    try:
        _wait(lambda: "evt_replay1" in _body_text(outbox),
              message="replayed history missing")
        text = _body_text(outbox)
        assert text.index(": connected") < text.index("evt_replay1")
    finally:
        disconnect()
        thread.join(timeout=5)


def test_stream_unsubscribes_on_disconnect(app):
    bus = app.state.bus
    thread, outbox, disconnect = _drive_stream(app)
    _wait(lambda: ": connected" in _body_text(outbox),
          message="stream never started")
    disconnect()
    thread.join(timeout=5)
    assert not thread.is_alive(), "stream did not shut down on disconnect"
    for topic in ("events.created", "events.updated", "camera.health", "alerts.updated"):
        assert not bus._handlers.get(topic), f"handler left subscribed on {topic}"


def test_stream_unknown_topic_falls_back_to_all(app):
    thread, outbox, disconnect = _drive_stream(app, query="topics=nope")
    try:
        start = _start_response(outbox)
        assert start["status"] == 200
        _wait(lambda: ": connected" in _body_text(outbox),
              message="stream never started")
    finally:
        disconnect()
        thread.join(timeout=5)


# -- camera health summary ------------------------------------------------


def test_health_summary_empty(client):
    response = client.get("/api/v1/cameras/health/summary")
    assert response.status_code == 200
    assert response.json() == []


def test_health_summary_statuses(client, app):
    from datetime import timedelta

    session = next(get_session())
    try:
        _seed_camera(session, "cam_on", enabled=True)
        _seed_camera(session, "cam_off_row", enabled=True)
        _seed_camera(session, "cam_disabled", enabled=False)
        _seed_camera(session, "cam_unknown", enabled=True)
        _seed_camera(session, "cam_gone", enabled=True, deleted=True)
        _seed_health(session, "cam_on", state="streaming", health="healthy",
                     ai_status="healthy", ts=utcnow())
        _seed_health(session, "cam_off_row", state="offline", health="offline",
                     ai_status="offline", ts=utcnow())
        _seed_health(session, "cam_disabled", state="streaming", health="healthy",
                     ai_status="healthy", ts=utcnow())
        _seed_health(session, "cam_unknown", state="streaming", health="degraded",
                     ai_status="degraded", ts=utcnow() - timedelta(minutes=10))
        session.commit()
    finally:
        session.close()

    response = client.get("/api/v1/cameras/health/summary")
    assert response.status_code == 200
    items = {i["camera_id"]: i for i in response.json()}
    assert "cam_gone" not in items
    assert items["cam_on"]["status"] == "online"
    assert items["cam_on"]["stale"] is False
    assert items["cam_off_row"]["status"] == "offline"
    assert items["cam_disabled"]["status"] == "disabled"
    assert items["cam_unknown"]["status"] == "degraded"
    assert items["cam_unknown"]["stale"] is True


def test_health_summary_status_mapping(client, app):
    session = next(get_session())
    try:
        _seed_camera(session, "cam_retry")
        _seed_camera(session, "cam_err")
        _seed_health(session, "cam_retry", state="reconnecting", health="degraded",
                     ai_status="offline", ts=utcnow())
        _seed_health(session, "cam_err", state="streaming", health="error",
                     ai_status="offline", ts=utcnow(), error="decode failed")
        session.commit()
    finally:
        session.close()

    items = {i["camera_id"]: i
             for i in client.get("/api/v1/cameras/health/summary").json()}
    assert items["cam_retry"]["status"] == "retrying"
    assert items["cam_err"]["status"] == "offline"
    assert items["cam_err"]["error"] == "decode failed"


def test_health_summary_requires_api_key_in_api_key_mode(api_key_client):
    response = api_key_client.get("/api/v1/cameras/health/summary")
    assert response.status_code == 401
    response = api_key_client.get(
        "/api/v1/cameras/health/summary", headers={"X-API-Key": "viewer-key"}
    )
    assert response.status_code == 200


# -- frame preview --------------------------------------------------------


def _install_fake_runner(app, camera_id="cam1", frame=...):
    if frame is ...:
        frame = SimpleNamespace(data=b"\xff\xd8fakejpeg", ts=utcnow())

    class FakeBuffer:
        def latest(self):
            return frame

    app.state.runner = SimpleNamespace(
        pipeline_settings=SimpleNamespace(camera_id=camera_id),
        buffer=FakeBuffer(),
    )
    return app


def test_preview_404_unknown_camera(client, app):
    _install_fake_runner(app, camera_id="nope")
    response = client.get("/api/v1/cameras/does-not-exist/preview.jpg")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_preview_409_without_runner(client, app, db_session):
    _seed_camera(db_session, "cam1")
    db_session.commit()
    response = client.get("/api/v1/cameras/cam1/preview.jpg")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "preview_unavailable"


def test_preview_409_when_runner_serves_another_camera(client, app, db_session):
    _seed_camera(db_session, "cam1")
    db_session.commit()
    _install_fake_runner(app, camera_id="cam2")
    response = client.get("/api/v1/cameras/cam1/preview.jpg")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "preview_unavailable"


def test_preview_409_when_buffer_empty(client, app, db_session):
    _seed_camera(db_session, "cam1")
    db_session.commit()
    _install_fake_runner(app, camera_id="cam1", frame=None)
    response = client.get("/api/v1/cameras/cam1/preview.jpg")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "preview_unavailable"


def test_preview_returns_jpeg_with_timestamp(client, app, db_session):
    _seed_camera(db_session, "cam1")
    db_session.commit()
    _install_fake_runner(app, camera_id="cam1")
    response = client.get("/api/v1/cameras/cam1/preview.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content == b"\xff\xd8fakejpeg"
    assert response.headers["x-frame-timestamp"]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-preview-mode"] == "preview"


# -- bus publishes on API activity ---------------------------------------


def test_event_status_transition_publishes_events_updated(client, app, db_session):
    _seed_camera(db_session, "cam1")
    _seed_event(db_session, "evt_pub1")
    db_session.commit()

    captured = []
    app.state.bus.subscribe("events.updated", captured.append)

    response = client.post(
        "/api/v1/events/evt_pub1/status", json={"status": "acknowledged"}
    )
    assert response.status_code == 200
    assert captured, "events.updated was not published"
    payload = captured[-1]
    assert payload["event_id"] == "evt_pub1"
    assert payload["status"] == "acknowledged"
    assert payload["actor"] == "dev-admin"


def test_alert_dispatcher_publishes_alerts_updated():
    from packages.common.bus import InMemoryBus
    from services.alerts.dispatcher import AlertDispatcher
    from services.alerts.types import AlertResult

    class FakeRouter:
        def dispatch(self, event):
            return [
                AlertResult(channel="in_app", status="sent", alert_id="alr_1"),
                AlertResult(channel="webhook", status="failed", reason="timeout"),
            ]

    bus = InMemoryBus()
    captured = []
    bus.subscribe("alerts.updated", captured.append)
    dispatcher = AlertDispatcher(FakeRouter(), queue_size=4, bus=bus)
    dispatcher.start()
    try:
        event = SimpleNamespace(event_id="evt_alert1")
        assert dispatcher.submit(event)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not captured:
            time.sleep(0.02)
    finally:
        dispatcher.stop(timeout=3.0)

    assert captured
    payload = captured[-1]
    assert payload["event_id"] == "evt_alert1"
    assert payload["results"][0]["channel"] == "in_app"
    assert payload["results"][0]["status"] == "sent"
    assert payload["results"][1]["status"] == "failed"


def test_flush_health_publishes_camera_health(db_session):
    from packages.common.bus import InMemoryBus
    from packages.db import models
    from packages.schemas.camera import CameraState
    from packages.schemas.common import HealthState
    from services.pipeline.runner import PipelineRunner

    _seed_camera(db_session, "cam1")
    db_session.commit()

    snapshot = SimpleNamespace(
        camera_id="cam1",
        state=CameraState.STREAMING,
        health=HealthState.HEALTHY,
        ai_status=HealthState.HEALTHY,
        fps=4.9,
        frame_drops=0,
        latency_ms=12.0,
        reconnect_count=0,
        frames_processed=42,
        last_frame_at=utcnow(),
        error=None,
        details={},
    )
    bus = InMemoryBus()
    captured = []
    bus.subscribe("camera.health", captured.append)

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.bus = bus
    runner.worker = SimpleNamespace(snapshot=lambda: snapshot)
    runner.flush_health()

    assert captured, "camera.health was not published"
    assert captured[-1]["camera_id"] == "cam1"
    assert captured[-1]["health"] == "healthy"
    row = db_session.query(models.CameraHealth).filter_by(camera_id="cam1").one()
    assert row.frames_processed == 42


def test_buffer_latest_returns_newest_frame():
    from datetime import timedelta

    from services.camera.types import FramePacket
    from services.evidence.buffer import RollingFrameBuffer

    buffer = RollingFrameBuffer("cam1", max_seconds=5.0)
    assert buffer.latest() is None
    base = utcnow()
    p1 = FramePacket(camera_id="cam1", frame_id=1, ts=base, data=b"first")
    p2 = FramePacket(camera_id="cam1", frame_id=2, ts=base + timedelta(seconds=1), data=b"second")
    buffer.append(p1)
    buffer.append(p2)
    latest = buffer.latest()
    assert latest is not None
    assert latest.data == b"second"
