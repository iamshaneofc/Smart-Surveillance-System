"""F2-M observability: unhandled errors are logged with traceback (secrets
redacted), and retention sweeps, alert dispatch, and evaluation runs emit logs.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from packages.db import base as db_base
from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus

NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
PAST = NOW - timedelta(days=2)


@pytest.fixture(autouse=True)
def _db(settings):
    db_base.dispose()
    db_base.configure(settings.database_url)
    db_base.create_all()
    yield
    db_base.dispose()


def _request():
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/v1/cameras",
        "raw_path": b"/api/v1/cameras",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    return Request(scope)


def test_unhandled_error_handler_logs_redacted_traceback(caplog):
    from apps.api.errors import unhandled_error_handler

    exc = RuntimeError("connection refused for rtsp://admin:secret123@10.0.0.1/stream")
    with caplog.at_level(logging.ERROR):
        response = asyncio.run(unhandled_error_handler(_request(), exc))

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "an unexpected error occurred"
    assert "details" in body["error"]

    records = [r for r in caplog.records if r.name == "apps.api.errors"]
    assert records, "unhandled error must be logged"
    message = records[0].getMessage()
    assert "unhandled error on GET /api/v1/cameras" in message
    assert "RuntimeError" in message
    assert "rtsp://***:***@" in message
    assert "secret123" not in message
    assert records[0].exc_info is not None, "log.exception must attach traceback"


def test_retention_sweep_logs_deletion(caplog, tmp_path):
    from services.evidence.retention import run_retention_sweep
    from services.evidence.store import LocalDiskEvidenceStore

    from tests.test_f2_retention import _seed_event, _seed_evidence

    store = LocalDiskEvidenceStore(tmp_path)
    store.save("cam1", "evt_ret1", "clip.mp4", b"data", "video/mp4")

    with db_base.session_scope() as session:
        _seed_event(session)
        _seed_evidence(session, expires_at=PAST)

    with caplog.at_level(logging.INFO):
        with db_base.session_scope() as session:
            report = run_retention_sweep(session, store, now=NOW)

    assert report.deleted == 1
    deleted = [r for r in caplog.records if r.getMessage() == "retention deleted evidence"]
    assert deleted, "retention deletion must be logged"
    assert deleted[0].evidence_id == "evd_ret1"


def test_evaluation_missing_manifest_logs_skip(tmp_path, caplog):
    from evaluation.runner import run_evaluation
    from services.inference.interfaces import default_registry

    detector = default_registry().create("null")
    with caplog.at_level(logging.WARNING):
        report = run_evaluation(tmp_path / "does-not-exist.json", detector)

    assert report["status"] == "EVALUATION DATASET NOT AVAILABLE"
    assert report["metrics"] is None
    skipped = [r for r in caplog.records if r.getMessage() == "evaluation skipped"]
    assert skipped, "unavailable evaluation must be logged"
    assert getattr(skipped[0], "reason", None) == "manifest missing"


def test_alert_send_failure_logs_and_redacts(caplog):
    from services.alerts.router import AlertRouter

    class FailingNotifier:
        channel = "boom"

        def send(self, message):
            raise ValueError("post failed for https://user:pass@example.com/hook")

    router = AlertRouter(notifiers={"boom": FailingNotifier()})
    event = Event(
        event_id="evt_log",
        camera_id="cam1",
        timestamp=NOW,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        status=EventStatus.RESOLVED,
        confidence=0.9,
        model_versions={"detector": "stub:1"},
        created_at=NOW,
        updated_at=NOW,
    )

    with caplog.at_level(logging.INFO):
        results = router.dispatch(event, channels=["boom"])

    assert results[0].status == "failed"
    assert "user:pass" not in results[0].reason
    assert "***:***@" in results[0].reason

    failed = [r for r in caplog.records if r.getMessage() == "alert send failed"]
    assert failed, "alert failure must be logged"
    assert failed[0].exc_info is not None
    assert getattr(failed[0], "channel", None) == "boom"
