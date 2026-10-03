import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from packages.config import Settings
from packages.db import base as db_base, models
from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus

ADMIN = {"X-API-Key": "admin-key"}
OPERATOR = {"X-API-Key": "operator-key"}
VIEWER = {"X-API-Key": "viewer-key"}

CONTENT = b"FAKE-JPEG-EVIDENCE-BYTES"
NOW = datetime(2026, 6, 15, 22, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        bus_url="memory://",
        auth_mode="disabled",
        rate_limit_per_minute=10000,
        log_level="WARNING",
        evidence={"root": str(tmp_path / "evd")},
    )


@pytest.fixture()
def api_key_settings(tmp_path) -> Settings:
    from packages.config import ApiKeySpec

    return Settings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'auth.db'}",
        bus_url="memory://",
        auth_mode="api_key",
        auth_api_keys=[
            ApiKeySpec(key="viewer-key", user="viewer1", roles=["viewer"]),
            ApiKeySpec(key="operator-key", user="operator1", roles=["operator"]),
            ApiKeySpec(key="admin-key", user="admin1", roles=["admin"]),
        ],
        rate_limit_per_minute=10000,
        log_level="WARNING",
        evidence={"root": str(tmp_path / "evd_auth")},
    )


def _write_file(root: str, uri: str, data: bytes) -> Path:
    path = Path(root) / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _seed_evidence(
    evidence_id="evd_dl1",
    uri="cam1/evt_dl1/snap.jpg",
    sha256=None,
    storage_backend="local",
    write_file=True,
    root=None,
):
    from services.events.repository import SqlEventRepository

    with db_base.session_scope() as session:
        SqlEventRepository(session).save(
            Event(
                event_id="evt_dl1",
                camera_id="cam1",
                timestamp=NOW,
                event_type="restricted_zone_intrusion",
                severity=Severity.HIGH,
                status=EventStatus.NEW,
                confidence=0.9,
                model_versions={},
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            models.Evidence(
                id=evidence_id,
                event_id="evt_dl1",
                camera_id="cam1",
                type="snapshot",
                uri=uri,
                sha256=sha256 if sha256 is not None else hashlib.sha256(CONTENT).hexdigest(),
                size_bytes=len(CONTENT),
                content_type="image/jpeg",
                captured_at=NOW,
                expires_at=None,
                storage_backend=storage_backend,
            )
        )
    if write_file and root:
        _write_file(root, uri, CONTENT)


def _audit_actions():
    from sqlalchemy import select

    with db_base.session_scope() as session:
        return list(session.execute(select(models.AuditLog)).scalars().all())


def test_download_returns_file_with_checksum(client, settings):
    _seed_evidence(root=settings.evidence.root)

    response = client.get("/api/v1/evidence/evd_dl1/download")
    assert response.status_code == 200
    assert response.content == CONTENT
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.headers["X-Checksum-SHA256"] == hashlib.sha256(CONTENT).hexdigest()
    assert 'filename="evd_dl1.jpg"' in response.headers["content-disposition"]

    actions = [a.action for a in _audit_actions()]
    assert "evidence.download" in actions


def test_download_missing_row_and_file(client, settings):
    missing_row = client.get("/api/v1/evidence/nope/download")
    assert missing_row.status_code == 404
    assert missing_row.json()["error"]["code"] == "not_found"

    _seed_evidence(evidence_id="evd_nofile", uri="cam1/evt_dl1/missing.jpg", write_file=False)
    missing_file = client.get("/api/v1/evidence/evd_nofile/download")
    assert missing_file.status_code == 404
    assert "missing.jpg" not in missing_file.text


def test_download_blocks_path_traversal(client, settings, tmp_path):
    secret = tmp_path / "outside-secret.txt"
    secret.write_bytes(b"TOP-SECRET")

    _seed_evidence(
        evidence_id="evd_evil",
        uri="../outside-secret.txt",
        sha256=hashlib.sha256(b"TOP-SECRET").hexdigest(),
        write_file=False,
    )
    response = client.get("/api/v1/evidence/evd_evil/download")
    assert response.status_code == 404
    assert b"TOP-SECRET" not in response.content
    assert "outside-secret" not in response.text


def test_download_integrity_mismatch(client, settings):
    _seed_evidence(
        evidence_id="evd_bad",
        uri="cam1/evt_dl1/bad.jpg",
        sha256="0" * 64,
        root=settings.evidence.root,
    )
    response = client.get("/api/v1/evidence/evd_bad/download")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "integrity_mismatch"
    assert b"FAKE-JPEG" not in response.content


def test_download_rejects_non_local_backend(client, settings):
    _seed_evidence(
        evidence_id="evd_mem",
        uri="memory://cam1/evt_dl1/snap.jpg",
        storage_backend="memory",
        write_file=False,
    )
    response = client.get("/api/v1/evidence/evd_mem/download")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "unsupported_backend"


def test_download_authz(api_key_client, api_key_settings):
    _seed_evidence(root=api_key_settings.evidence.root)

    denied = api_key_client.get("/api/v1/evidence/evd_dl1/download", headers=VIEWER)
    assert denied.status_code == 403
    denied = api_key_client.get("/api/v1/evidence/evd_dl1/download", headers=OPERATOR)
    assert denied.status_code == 403

    allowed = api_key_client.get("/api/v1/evidence/evd_dl1/download", headers=ADMIN)
    assert allowed.status_code == 200
    assert allowed.content == CONTENT
