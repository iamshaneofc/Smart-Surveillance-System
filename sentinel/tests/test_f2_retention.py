from datetime import datetime, timedelta, timezone

import pytest

from packages.db import base as db_base, models
from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus
from services.evidence.retention import run_retention_sweep
from services.evidence.store import EvidenceStore, LocalDiskEvidenceStore

NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
PAST = NOW - timedelta(days=2)
FUTURE = NOW + timedelta(days=30)


@pytest.fixture(autouse=True)
def _db(settings):
    db_base.dispose()
    db_base.configure(settings.database_url)
    db_base.create_all()
    yield
    db_base.dispose()


class ExplodingStore(EvidenceStore):
    backend = "local"

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def exists(self, uri: str) -> bool:
        return True

    def delete(self, uri: str) -> None:
        raise OSError("disk failure")


def _seed_event(session, event_id="evt_ret1", status=EventStatus.RESOLVED):
    event = Event(
        event_id=event_id,
        camera_id="cam1",
        timestamp=NOW,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        status=status,
        confidence=0.9,
        model_versions={"detector": "stub:1"},
        created_at=NOW,
        updated_at=NOW,
    )
    from services.events.repository import SqlEventRepository

    SqlEventRepository(session).save(event)
    return event


def _seed_evidence(
    session,
    evidence_id="evd_ret1",
    event_id="evt_ret1",
    expires_at=PAST,
    uri="cam1/evt_ret1/clip.mp4",
):
    row = models.Evidence(
        id=evidence_id,
        event_id=event_id,
        camera_id="cam1",
        type="clip",
        uri=uri,
        sha256="b" * 64,
        size_bytes=10,
        content_type="video/mp4",
        captured_at=NOW,
        expires_at=expires_at,
        storage_backend="local",
    )
    session.add(row)
    session.flush()
    return row


def _audit_actions():
    from sqlalchemy import select

    with db_base.session_scope() as session:
        return list(session.execute(select(models.AuditLog)).scalars().all())


def test_sweep_deletes_expired_evidence(tmp_path):
    store = LocalDiskEvidenceStore(tmp_path)
    store.save("cam1", "evt_ret1", "clip.mp4", b"data", "video/mp4")

    with db_base.session_scope() as session:
        _seed_event(session)
        _seed_evidence(session)

    with db_base.session_scope() as session:
        report = run_retention_sweep(session, store, now=NOW)

    assert report.candidates == 1
    assert report.deleted == 1
    assert report.failed == 0
    assert not (tmp_path / "cam1" / "evt_ret1" / "clip.mp4").exists()

    with db_base.session_scope() as session:
        assert session.get(models.Evidence, "evd_ret1") is None

    audits = _audit_actions()
    deletes = [a for a in audits if a.action == "evidence.retention_delete"]
    assert len(deletes) == 1
    assert deletes[0].details["reason"] == "expired"
    assert deletes[0].details["event_id"] == "evt_ret1"
    assert deletes[0].actor == "system:retention"


def test_sweep_is_idempotent(tmp_path):
    store = LocalDiskEvidenceStore(tmp_path)
    store.save("cam1", "evt_ret1", "clip.mp4", b"data", "video/mp4")

    with db_base.session_scope() as session:
        _seed_event(session)
        _seed_evidence(session)

    with db_base.session_scope() as session:
        first = run_retention_sweep(session, store, now=NOW)
    with db_base.session_scope() as session:
        second = run_retention_sweep(session, store, now=NOW)

    assert first.deleted == 1
    assert second.candidates == 0
    assert second.deleted == 0


def test_sweep_missing_file_deletes_metadata_only(tmp_path):
    store = LocalDiskEvidenceStore(tmp_path)

    with db_base.session_scope() as session:
        _seed_event(session)
        _seed_evidence(session, uri="cam1/evt_ret1/gone.mp4")

    with db_base.session_scope() as session:
        report = run_retention_sweep(session, store, now=NOW)

    assert report.deleted == 1
    assert report.missing_files == 1

    with db_base.session_scope() as session:
        assert session.get(models.Evidence, "evd_ret1") is None

    audits = [a for a in _audit_actions() if a.action == "evidence.retention_delete"]
    assert audits[0].details["reason"] == "file_missing"


def test_sweep_skips_active_events_unless_allowed(tmp_path):
    store = LocalDiskEvidenceStore(tmp_path)
    store.save("cam1", "evt_ret1", "clip.mp4", b"data", "video/mp4")

    with db_base.session_scope() as session:
        _seed_event(session, status=EventStatus.NEW)
        _seed_evidence(session)

    with db_base.session_scope() as session:
        report = run_retention_sweep(session, store, now=NOW)
    assert report.skipped_active == 1
    assert report.deleted == 0
    with db_base.session_scope() as session:
        assert session.get(models.Evidence, "evd_ret1") is not None

    with db_base.session_scope() as session:
        allowed = run_retention_sweep(
            session, store, allow_active_event_deletion=True, now=NOW
        )
    assert allowed.deleted == 1
    with db_base.session_scope() as session:
        assert session.get(models.Evidence, "evd_ret1") is None


def test_sweep_failure_keeps_row_and_audits(tmp_path):
    store = ExplodingStore()

    with db_base.session_scope() as session:
        _seed_event(session)
        _seed_evidence(session)

    with db_base.session_scope() as session:
        report = run_retention_sweep(session, store, now=NOW)

    assert report.failed == 1
    assert report.deleted == 0
    with db_base.session_scope() as session:
        assert session.get(models.Evidence, "evd_ret1") is not None

    failures = [a for a in _audit_actions() if a.action == "evidence.retention_failed"]
    assert len(failures) == 1
    assert "disk failure" in failures[0].details["error"]


def test_dry_run_changes_nothing(tmp_path):
    store = LocalDiskEvidenceStore(tmp_path)
    store.save("cam1", "evt_ret1", "clip.mp4", b"data", "video/mp4")

    with db_base.session_scope() as session:
        _seed_event(session)
        _seed_evidence(session)

    with db_base.session_scope() as session:
        report = run_retention_sweep(session, store, now=NOW, dry_run=True)

    assert report.dry_run is True
    assert report.candidates == 1
    assert report.deleted == 0
    assert (tmp_path / "cam1" / "evt_ret1" / "clip.mp4").exists()
    with db_base.session_scope() as session:
        assert session.get(models.Evidence, "evd_ret1") is not None
    audits = [a for a in _audit_actions() if a.action.startswith("evidence.retention")]
    assert audits == []


def test_sweep_ignores_unexpired_and_null_expiry(tmp_path):
    store = LocalDiskEvidenceStore(tmp_path)

    with db_base.session_scope() as session:
        _seed_event(session)
        _seed_evidence(session, evidence_id="evd_future", expires_at=FUTURE)
        _seed_evidence(
            session, evidence_id="evd_null", expires_at=None, uri="cam1/evt_ret1/keep.mp4"
        )

    with db_base.session_scope() as session:
        report = run_retention_sweep(session, store, now=NOW)

    assert report.candidates == 0
    with db_base.session_scope() as session:
        assert session.get(models.Evidence, "evd_future") is not None
        assert session.get(models.Evidence, "evd_null") is not None


def test_retention_settings_flag():
    from packages.config import Settings

    defaults = Settings(
        env="test",
        database_url="sqlite://",
        bus_url="memory://",
        auth_mode="disabled",
        rate_limit_per_minute=10000,
        log_level="WARNING",
    )
    assert defaults.evidence.allow_active_event_deletion is False

    overridden = Settings(
        env="test",
        database_url="sqlite://",
        bus_url="memory://",
        auth_mode="disabled",
        rate_limit_per_minute=10000,
        log_level="WARNING",
        evidence={"root": "./var/evidence", "allow_active_event_deletion": True},
    )
    assert overridden.evidence.allow_active_event_deletion is True
