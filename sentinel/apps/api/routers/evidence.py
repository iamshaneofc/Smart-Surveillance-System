import hashlib
import re
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from apps.api.deps import SettingsDep, get_db_session, require_permission
from apps.api.errors import AppError
from packages.common.logging import get_logger
from packages.db import models
from packages.db.audit import record_audit
from packages.schemas.auth import Principal
from packages.schemas.common import Page
from packages.schemas.evidence import EvidenceItem, EvidenceType

log = get_logger(__name__)

router = APIRouter(prefix="/evidence", tags=["evidence"])

CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
}


def _evidence_item(row: models.Evidence) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=row.id,
        event_id=row.event_id,
        camera_id=row.camera_id,
        type=EvidenceType(row.type),
        uri=row.uri,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        content_type=row.content_type,
        width=row.width,
        height=row.height,
        duration_ms=row.duration_ms,
        captured_at=row.captured_at,
        expires_at=row.expires_at,
        storage_backend=row.storage_backend,
        metadata=row.metadata_ or {},
        created_at=row.created_at,
    )


@router.get("", response_model=Page[EvidenceItem])
def list_evidence(
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("evidence:read")),
    event_id: str | None = None,
    camera_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    stmt = select(models.Evidence)
    if event_id:
        stmt = stmt.where(models.Evidence.event_id == event_id)
    if camera_id:
        stmt = stmt.where(models.Evidence.camera_id == camera_id)
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(
        stmt.order_by(models.Evidence.captured_at.desc()).limit(limit).offset(offset)
    ).scalars()
    return Page[EvidenceItem](
        items=[_evidence_item(r) for r in rows], total=int(total), limit=limit, offset=offset
    )


@router.get("/{evidence_id}", response_model=EvidenceItem)
def get_evidence(
    evidence_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("evidence:read")),
):
    row = session.get(models.Evidence, evidence_id)
    if row is None:
        raise AppError("not_found", f"evidence '{evidence_id}' not found", 404)
    return _evidence_item(row)


@router.get("/{evidence_id}/download")
def download_evidence(
    evidence_id: str,
    settings: SettingsDep,
    session: Session = Depends(get_db_session),
    principal: Principal = Depends(require_permission("evidence:export")),
):
    row = session.get(models.Evidence, evidence_id)
    if row is None:
        raise AppError("not_found", f"evidence '{evidence_id}' not found", 404)
    if row.storage_backend != "local":
        raise AppError(
            "unsupported_backend",
            f"storage backend '{row.storage_backend}' cannot be downloaded directly",
            409,
        )

    root = Path(settings.evidence.root).resolve()
    resolved = (root / row.uri).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        log.warning(
            "evidence download file missing",
            extra={"evidence_id": evidence_id, "event_id": row.event_id},
        )
        raise AppError("not_found", "evidence file not found", 404)

    data = resolved.read_bytes()
    if row.sha256:
        actual = hashlib.sha256(data).hexdigest()
        if actual != row.sha256:
            log.error(
                "evidence integrity mismatch",
                extra={"evidence_id": evidence_id, "expected": row.sha256},
            )
            raise AppError(
                "integrity_mismatch",
                "stored checksum does not match the file on disk",
                409,
            )

    ext = CONTENT_TYPE_EXTENSIONS.get(row.content_type or "")
    if not ext:
        suffix = Path(row.uri).suffix.lower()
        ext = suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ".bin"

    record_audit(
        session,
        principal,
        action="evidence.download",
        resource_type="evidence",
        resource_id=evidence_id,
        details={"event_id": row.event_id, "type": row.type, "size_bytes": row.size_bytes},
    )
    session.flush()

    headers = {"Content-Disposition": f'attachment; filename="{row.id}{ext}"'}
    if row.sha256:
        headers["X-Checksum-SHA256"] = row.sha256
    return Response(
        content=data,
        media_type=row.content_type or "application/octet-stream",
        headers=headers,
    )
