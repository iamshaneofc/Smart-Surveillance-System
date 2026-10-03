import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from packages.db import base as db_base, models
from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus
from services.alerts.dispatcher import AlertDispatcher
from services.alerts.notifiers import (
    DatabaseNotifier,
    LoggingNotifier,
    WebhookNotifier,
)
from services.alerts.router import AlertRouter

TZ = timezone.utc
ADMIN = {"X-API-Key": "admin-key"}
OPERATOR = {"X-API-Key": "operator-key"}
VIEWER = {"X-API-Key": "viewer-key"}


@pytest.fixture(autouse=True)
def _db(settings):
    db_base.dispose()
    db_base.configure(settings.database_url)
    db_base.create_all()
    yield
    db_base.dispose()


def _event(ts=None, event_id="evt_a1", summary="", evidence_ids=None):
    ts = ts or datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    return Event(
        event_id=event_id,
        camera_id="cam1",
        timestamp=ts,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        status=EventStatus.NEW,
        confidence=0.9,
        summary=summary,
        evidence_ids=evidence_ids or [],
        created_at=ts,
        updated_at=ts,
    )


def start_webhook_server(status_script=None, delay=0.0):
    """Local webhook receiver. status_script: list of codes popped per request."""
    state = {"requests": [], "lock": threading.Lock()}
    script = list(status_script or [])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            with state["lock"]:
                state["requests"].append(body)
            if delay:
                time.sleep(delay)
            with state["lock"]:
                status = script.pop(0) if script else 200
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/hook"
    return state, url, server


def test_webhook_delivery_sends_required_payload():
    state, url, server = start_webhook_server()
    try:
        notifier = WebhookNotifier(url, timeout=2.0, max_retries=0)
        router = AlertRouter(notifiers={"webhook": notifier}, default_channels=["webhook"])
        event = _event(summary="person remained inside for 1.9 seconds", evidence_ids=["evd1", "evd2"])
        results = router.dispatch(event)

        assert results[0].status == "sent"
        assert len(state["requests"]) == 1
        payload = json.loads(state["requests"][0])
        assert payload["event_id"] == "evt_a1"
        assert payload["event_type"] == "restricted_zone_intrusion"
        assert payload["severity"] == "high"
        assert payload["camera_id"] == "cam1"
        assert payload["timestamp"] == event.timestamp.isoformat()
        assert payload["summary"] == "person remained inside for 1.9 seconds"
        assert payload["evidence_ids"] == ["evd1", "evd2"]
    finally:
        server.shutdown()


def test_webhook_retries_then_succeeds():
    state, url, server = start_webhook_server(status_script=[500, 500, 200])
    try:
        notifier = WebhookNotifier(
            url, timeout=2.0, max_retries=2, backoff_seconds=0.0, sleep=lambda s: None
        )
        router = AlertRouter(notifiers={"webhook": notifier}, default_channels=["webhook"])
        results = router.dispatch(_event())
        assert results[0].status == "sent"
        assert len(state["requests"]) == 3
    finally:
        server.shutdown()


def test_webhook_permanent_failure_is_bounded_and_reported():
    state, url, server = start_webhook_server(status_script=[500, 500, 500])
    try:
        notifier = WebhookNotifier(
            url, timeout=2.0, max_retries=2, backoff_seconds=0.0, sleep=lambda s: None
        )
        router = AlertRouter(notifiers={"webhook": notifier}, default_channels=["webhook"])
        results = router.dispatch(_event())
        assert results[0].status == "failed"
        assert "500" in results[0].reason
        assert len(state["requests"]) == 3  # 1 attempt + 2 retries, then gives up
    finally:
        server.shutdown()


def test_webhook_timeout_is_bounded():
    state, url, server = start_webhook_server(delay=0.6)
    try:
        notifier = WebhookNotifier(
            url, timeout=0.15, max_retries=1, backoff_seconds=0.0, sleep=lambda s: None
        )
        router = AlertRouter(notifiers={"webhook": notifier}, default_channels=["webhook"])
        results = router.dispatch(_event())
        assert results[0].status == "failed"
        assert len(state["requests"]) == 2  # 1 attempt + 1 retry
    finally:
        server.shutdown()


def test_webhook_failure_reason_redacts_credentials():
    state, url, server = start_webhook_server(status_script=[500])
    try:
        credentialed = url.replace("http://", "http://user:supersecret@")
        notifier = WebhookNotifier(
            credentialed, timeout=2.0, max_retries=0, sleep=lambda s: None
        )
        router = AlertRouter(notifiers={"webhook": notifier}, default_channels=["webhook"])
        results = router.dispatch(_event())
        assert results[0].status == "failed"
        assert "supersecret" not in results[0].reason
        assert "user:supersecret" not in str(results)
    finally:
        server.shutdown()


def test_webhook_rejects_non_http_url():
    with pytest.raises(ValueError):
        WebhookNotifier("file:///etc/passwd")


def test_duplicate_event_dispatch_is_idempotent():
    notifier = LoggingNotifier()
    router = AlertRouter(notifiers={"dashboard": notifier}, default_channels=["dashboard"])
    ts = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)

    first = router.dispatch(_event(ts, event_id="evt_dup"))
    second = router.dispatch(_event(ts, event_id="evt_dup"))

    assert first[0].status == "sent"
    assert second[0].status == "skipped"
    assert second[0].reason == "duplicate"
    assert len(notifier.sent) == 1


def test_database_notifier_persists_alert_row():
    notifier = DatabaseNotifier()
    router = AlertRouter(notifiers={"in_app": notifier}, default_channels=["in_app"])
    results = router.dispatch(_event(summary="s", evidence_ids=["evd1"]))
    assert results[0].status == "sent"

    with db_base.session_scope() as session:
        from sqlalchemy import select

        rows = session.execute(select(models.Alert)).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_id == "evt_a1"
    assert rows[0].channel == "in_app"
    assert rows[0].status == "sent"
    assert rows[0].attempts == 1
    assert rows[0].sent_at is not None


def test_dispatcher_survives_router_failure():
    class BrokenRouter:
        def dispatch(self, event):
            raise RuntimeError("dispatch exploded")

    dispatcher = AlertDispatcher(BrokenRouter(), queue_size=10)
    dispatcher.start()
    try:
        assert dispatcher.submit(_event(event_id="evt_x1")) is True
        deadline = time.time() + 2.0
        while dispatcher.processed < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert dispatcher.processed == 1
        assert dispatcher.failed == 1

        assert dispatcher.submit(_event(event_id="evt_x2")) is True
        deadline = time.time() + 2.0
        while dispatcher.processed < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert dispatcher.processed == 2  # worker thread is still alive
    finally:
        dispatcher.stop()


def test_dispatcher_queue_full_drops_without_blocking():
    dispatcher = AlertDispatcher(AlertRouter(notifiers={}), queue_size=1)
    assert dispatcher.submit(_event(event_id="evt_q1")) is True
    assert dispatcher.submit(_event(event_id="evt_q2")) is False
    assert dispatcher.dropped == 1
    dispatcher.stop()


def test_pipeline_survives_alert_submit_failure():
    from tests.test_pipeline import HAS_CV2, JPG, build, feed, frame, scripted_where

    pipeline, engine, store, evidence = build(scripted_where((5, None)))

    class ExplodingDispatcher:
        def submit(self, event):
            raise RuntimeError("alert queue broken")

    pipeline.alert_dispatcher = ExplodingDispatcher()
    events = feed(pipeline, [frame(i, JPG if HAS_CV2 else b"f") for i in range(21)])
    assert len(events) == 1  # event still produced; alert failure did not crash


def test_alert_settings_defaults():
    from packages.config import Settings

    defaults = Settings(
        env="test",
        database_url="sqlite://",
        bus_url="memory://",
        auth_mode="disabled",
        rate_limit_per_minute=10000,
        log_level="WARNING",
    )
    assert defaults.alerts.enabled is False
    assert defaults.alerts.channels == ["in_app"]
    assert defaults.alerts.webhook_max_retries == 2
    assert defaults.alerts.queue_max == 100


def test_get_alerts_api(client):
    with db_base.session_scope() as session:
        session.add(
            models.Alert(
                event_id="evt_api_alert", channel="in_app", status="sent",
                attempts=1, sent_at=datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ),
            )
        )
        session.add(
            models.Alert(event_id="evt_api_alert", channel="webhook", status="failed", attempts=3)
        )

    listed = client.get("/api/v1/alerts")
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 2

    filtered = client.get("/api/v1/alerts", params={"channel": "webhook"})
    assert filtered.json()["total"] == 1
    item = filtered.json()["items"][0]
    assert item["status"] == "failed"
    assert item["attempts"] == 3

    filtered = client.get("/api/v1/alerts", params={"event_id": "evt_missing"})
    assert filtered.json()["total"] == 0


def test_alerts_api_authz(api_key_client):
    with db_base.session_scope() as session:
        session.add(
            models.Alert(event_id="evt_authz", channel="in_app", status="sent", attempts=1)
        )

    denied = api_key_client.get("/api/v1/alerts", headers=VIEWER)
    assert denied.status_code == 403

    allowed = api_key_client.get("/api/v1/alerts", headers=OPERATOR)
    assert allowed.status_code == 200
    assert allowed.json()["total"] == 1

    admin = api_key_client.get("/api/v1/alerts", headers=ADMIN)
    assert admin.status_code == 200
