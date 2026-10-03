from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from packages.db.audit import record_audit
from apps.api.deps import get_db_session, require_permission
from apps.api.errors import AppError
from packages.common.timeutil import utcnow
from packages.db import models
from packages.schemas.camera import (
    CameraCreate,
    CameraHealthSnapshot,
    CameraHealthSummary,
    CameraOut,
    CameraSourceType,
    CameraUpdate,
    RetentionPolicy,
    _validate_source,
)
from packages.schemas.common import HealthState, Page
from packages.schemas.event import EventStatus

router = APIRouter(prefix="/cameras", tags=["cameras"])

HEALTH_STALE_SECONDS = 60


def _camera_out(row: models.Camera) -> CameraOut:
    retention = RetentionPolicy(**(row.retention or {}))
    return CameraOut(
        id=row.id,
        camera_id=row.camera_id,
        name=row.name,
        location=row.location,
        site_id=row.site_id,
        source_type=row.source_type,
        enabled=row.enabled,
        detection_enabled=row.detection_enabled,
        recording_enabled=row.recording_enabled,
        detection_fps=row.detection_fps,
        width=row.width,
        height=row.height,
        timezone=row.timezone,
        model_profile=row.model_profile,
        rule_profile=row.rule_profile,
        retention=retention,
        metadata=row.metadata_ or {},
        stream_url_set=bool(row.stream_url),
        deleted_at=row.deleted_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _get_camera(
    session: Session, camera_id: str, include_deleted: bool = False
) -> models.Camera:
    stmt = select(models.Camera).where(models.Camera.camera_id == camera_id)
    if not include_deleted:
        stmt = stmt.where(models.Camera.deleted_at.is_(None))
    row = session.execute(stmt).scalar_one_or_none()
    if row is None:
        raise AppError("not_found", f"camera '{camera_id}' not found", 404)
    return row


def _validate_site(session: Session, site_id: str | None) -> None:
    if site_id and session.get(models.Site, site_id) is None:
        raise AppError("invalid_site", f"site '{site_id}' not found", 422)


@router.get("", response_model=Page[CameraOut])
def list_cameras(
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:read")),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    enabled: bool | None = None,
    site_id: str | None = None,
    include_deleted: bool = Query(default=False),
):
    stmt = select(models.Camera)
    if not include_deleted:
        stmt = stmt.where(models.Camera.deleted_at.is_(None))
    if enabled is not None:
        stmt = stmt.where(models.Camera.enabled == enabled)
    if site_id is not None:
        stmt = stmt.where(models.Camera.site_id == site_id)
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(
        stmt.order_by(models.Camera.camera_id).limit(limit).offset(offset)
    ).scalars()
    return Page[CameraOut](
        items=[_camera_out(r) for r in rows], total=int(total), limit=limit, offset=offset
    )


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(
    camera_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:read")),
    include_deleted: bool = Query(default=False),
):
    row = _get_camera(session, camera_id, include_deleted=include_deleted)
    return _camera_out(row)


@router.post("", status_code=201, response_model=CameraOut)
def create_camera(
    payload: CameraCreate,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:manage")),
):
    existing = session.execute(
        select(models.Camera).where(models.Camera.camera_id == payload.camera_id)
    ).scalar_one_or_none()
    if existing is not None:
        raise AppError(
            "conflict", f"camera '{payload.camera_id}' already exists", 409
        )
    _validate_site(session, payload.site_id)

    row = models.Camera(
        camera_id=payload.camera_id,
        name=payload.name,
        location=payload.location,
        site_id=payload.site_id,
        source_type=payload.source_type.value,
        stream_url=payload.stream_url,
        enabled=payload.enabled,
        detection_enabled=payload.detection_enabled,
        recording_enabled=payload.recording_enabled,
        detection_fps=payload.detection_fps,
        width=payload.width,
        height=payload.height,
        timezone=payload.timezone,
        model_profile=payload.model_profile,
        rule_profile=payload.rule_profile,
        retention=payload.retention.model_dump(),
        metadata_=payload.metadata,
    )
    session.add(row)
    record_audit(
        session,
        principal,
        action="camera.create",
        resource_type="camera",
        resource_id=payload.camera_id,
        details={
            "source_type": payload.source_type.value,
            "site_id": payload.site_id,
            "enabled": payload.enabled,
        },
    )
    session.flush()
    return _camera_out(row)


@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(
    camera_id: str,
    payload: CameraUpdate,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:manage")),
):
    row = _get_camera(session, camera_id)

    if payload.stream_url is not None:
        try:
            _validate_source(CameraSourceType(row.source_type), payload.stream_url)
        except ValueError as exc:
            raise AppError("invalid_stream_url", str(exc), 422) from exc
        row.stream_url = payload.stream_url

    changed: list[str] = []
    data = payload.model_dump(exclude_unset=True)
    data.pop("stream_url", None)
    for field, value in data.items():
        if field == "retention":
            value = value.model_dump()
            attr = "retention"
        else:
            attr = "metadata_" if field == "metadata" else field
        if getattr(row, attr) != value:
            setattr(row, attr, value)
            changed.append(field)

    record_audit(
        session,
        principal,
        action="camera.update",
        resource_type="camera",
        resource_id=camera_id,
        details={"changed": changed, "stream_url_changed": payload.stream_url is not None},
    )
    session.flush()
    return _camera_out(row)


@router.delete("/{camera_id}", status_code=204)
def delete_camera(
    camera_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:manage")),
):
    row = _get_camera(session, camera_id)
    row.deleted_at = datetime.now(timezone.utc)
    row.enabled = False
    row.detection_enabled = False
    record_audit(
        session,
        principal,
        action="camera.delete",
        resource_type="camera",
        resource_id=camera_id,
        details={
            "mode": "soft_delete",
            "note": "camera rows are soft-deleted so historical events and evidence stay valid",
        },
    )
    session.flush()


@router.get("/{camera_id}/health", response_model=list[CameraHealthSnapshot])
def get_camera_health(
    camera_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:read")),
    limit: int = Query(default=50, ge=1, le=500),
):
    rows = session.execute(
        select(models.CameraHealth)
        .where(models.CameraHealth.camera_id == camera_id)
        .order_by(models.CameraHealth.ts.desc())
        .limit(limit)
    ).scalars()
    return [
        CameraHealthSnapshot(
            camera_id=r.camera_id,
            state=r.state,
            health=HealthState(r.health),
            fps=r.fps,
            frame_drops=r.frame_drops,
            latency_ms=r.latency_ms,
            reconnect_count=r.reconnect_count,
            frames_processed=r.frames_processed,
            last_frame_at=r.last_frame_at,
            error=r.error,
            ai_status=HealthState(r.ai_status),
            ts=r.ts,
            details=r.details or {},
        )
        for r in rows
    ]


def _derive_status(enabled: bool, health_row: models.CameraHealth | None) -> str:
    if not enabled:
        return "disabled"
    if health_row is None:
        return "unknown"
    if health_row.state in ("offline", "stopped") or health_row.health in (
        "offline",
        "error",
    ):
        return "offline"
    if health_row.state in ("connecting", "reconnecting"):
        return "retrying"
    if health_row.health == "degraded" or health_row.ai_status == "degraded":
        return "degraded"
    if health_row.health == "healthy":
        return "online"
    return "unknown"


@router.get("/health/summary", response_model=list[CameraHealthSummary])
def camera_health_summary(
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:read")),
):
    """Latest health row per (non-deleted) camera with a derived status."""
    cameras = session.execute(
        select(models.Camera)
        .where(models.Camera.deleted_at.is_(None))
        .order_by(models.Camera.camera_id)
    ).scalars().all()

    rn = (
        func.row_number()
        .over(
            partition_by=models.CameraHealth.camera_id,
            order_by=models.CameraHealth.ts.desc(),
        )
        .label("rn")
    )
    subq = select(models.CameraHealth.id, rn).subquery()
    latest_ids = session.execute(
        select(subq.c.id).where(subq.c.rn == 1)
    ).scalars().all()
    health_by_camera: dict[str, models.CameraHealth] = {}
    if latest_ids:
        for row in session.execute(
            select(models.CameraHealth).where(models.CameraHealth.id.in_(latest_ids))
        ).scalars():
            health_by_camera[row.camera_id] = row

    now = utcnow()
    items: list[CameraHealthSummary] = []
    for cam in cameras:
        row = health_by_camera.get(cam.camera_id)
        health_ts = row.ts if row is not None else None
        if health_ts is not None and health_ts.tzinfo is None:
            health_ts = health_ts.replace(tzinfo=timezone.utc)
        stale = health_ts is None or (now - health_ts) > timedelta(seconds=HEALTH_STALE_SECONDS)
        items.append(
            CameraHealthSummary(
                camera_id=cam.camera_id,
                name=cam.name,
                location=cam.location,
                site_id=cam.site_id,
                enabled=cam.enabled,
                status=_derive_status(cam.enabled, row),
                stale=stale,
                state=row.state if row else None,
                health=row.health if row else None,
                ai_status=row.ai_status if row else None,
                fps=row.fps if row else None,
                frame_drops=row.frame_drops if row else None,
                reconnect_count=row.reconnect_count if row else None,
                last_frame_at=row.last_frame_at if row else None,
                error=row.error if row else None,
                health_ts=health_ts,
            )
        )
    return items


@router.get("/{camera_id}/preview.jpg")
def camera_preview(
    camera_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("cameras:read")),
):
    """Development frame preview from the in-process pipeline buffer.

    Honest contract: 200 = buffered preview frame (with its timestamp);
    409 preview_unavailable = camera exists but no live pipeline is feeding
    it. This is NOT a live stream and must be labelled PREVIEW by clients.
    """
    _get_camera(session, camera_id)
    runner = getattr(request.app.state, "runner", None)
    if runner is None or runner.pipeline_settings.camera_id != camera_id:
        raise AppError(
            "preview_unavailable",
            "no live pipeline is running for this camera",
            409,
        )
    frame = runner.buffer.latest()
    if frame is None:
        raise AppError("preview_unavailable", "no frames buffered yet", 409)
    return Response(
        content=frame.data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Timestamp": frame.ts.isoformat(),
            "X-Preview-Mode": "preview",
        },
    )
