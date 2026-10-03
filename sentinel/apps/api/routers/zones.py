from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from packages.db.audit import record_audit
from apps.api.deps import get_db_session, require_permission
from apps.api.errors import AppError
from packages.db import models
from packages.schemas.common import Page
from packages.schemas.rule import Zone, ZoneCreate, ZoneUpdate
from services.rules.geometry import is_simple_polygon

zones_router = APIRouter(prefix="/zones", tags=["zones"])
camera_zones_router = APIRouter(prefix="/cameras/{camera_id}/zones", tags=["zones"])


def _require_simple_polygon(polygon: list) -> None:
    if not is_simple_polygon([tuple(p) for p in polygon]):
        raise AppError(
            "invalid_polygon",
            "polygon must be simple: at least 3 points and no self-intersections",
            422,
        )


def _zone_out(row: models.Zone, camera_key: str) -> Zone:
    return Zone(
        id=row.id,
        camera_id=camera_key,
        name=row.name,
        zone_type=row.zone_type,
        polygon=row.polygon or [],
        anchor=row.anchor or "center",
        enabled=row.enabled,
        metadata=row.metadata_ or {},
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _get_zone(session: Session, zone_id: str) -> tuple[models.Zone, str]:
    row = session.get(models.Zone, zone_id)
    if row is None:
        raise AppError("not_found", f"zone '{zone_id}' not found", 404)
    camera = session.get(models.Camera, row.camera_id)
    camera_key = camera.camera_id if camera else row.camera_id
    return row, camera_key


def _get_camera(session: Session, camera_id: str) -> models.Camera:
    row = session.execute(
        select(models.Camera).where(
            models.Camera.camera_id == camera_id,
            models.Camera.deleted_at.is_(None),
        )
    ).scalar_one_or_none()
    if row is None:
        raise AppError("not_found", f"camera '{camera_id}' not found", 404)
    return row


def _check_unique_name(session: Session, camera_pk: str, name: str, exclude_zone: str | None = None):
    stmt = select(models.Zone).where(
        models.Zone.camera_id == camera_pk, models.Zone.name == name
    )
    if exclude_zone:
        stmt = stmt.where(models.Zone.id != exclude_zone)
    if session.execute(stmt).scalar_one_or_none() is not None:
        raise AppError(
            "conflict", f"zone '{name}' already exists on this camera", 409
        )


@camera_zones_router.get("", response_model=Page[Zone])
def list_camera_zones(
    camera_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("zones:read")),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    enabled: bool | None = None,
):
    camera = _get_camera(session, camera_id)
    stmt = select(models.Zone).where(models.Zone.camera_id == camera.id)
    if enabled is not None:
        stmt = stmt.where(models.Zone.enabled == enabled)
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(
        stmt.order_by(models.Zone.name).limit(limit).offset(offset)
    ).scalars()
    return Page[Zone](
        items=[_zone_out(r, camera.camera_id) for r in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@camera_zones_router.post("", status_code=201, response_model=Zone)
def create_zone(
    camera_id: str,
    payload: ZoneCreate,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("zones:manage")),
):
    camera = _get_camera(session, camera_id)
    _require_simple_polygon(payload.polygon)
    _check_unique_name(session, camera.id, payload.name)

    row = models.Zone(
        camera_id=camera.id,
        name=payload.name,
        zone_type=payload.zone_type.value,
        polygon=[list(p) for p in payload.polygon],
        anchor=payload.anchor,
        enabled=payload.enabled,
        metadata_=payload.metadata,
    )
    session.add(row)
    record_audit(
        session,
        principal,
        action="zone.create",
        resource_type="zone",
        resource_id=payload.name,
        details={"camera_id": camera_id, "zone_type": payload.zone_type.value},
    )
    session.flush()
    return _zone_out(row, camera.camera_id)


@zones_router.get("", response_model=Page[Zone])
def list_zones(
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("zones:read")),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    camera_id: str | None = None,
    enabled: bool | None = None,
):
    stmt = select(models.Zone, models.Camera.camera_id).join(
        models.Camera, models.Zone.camera_id == models.Camera.id
    )
    if camera_id is not None:
        stmt = stmt.where(models.Camera.camera_id == camera_id)
    if enabled is not None:
        stmt = stmt.where(models.Zone.enabled == enabled)
    total = session.execute(
        select(func.count()).select_from(stmt.subquery())
    ).scalar_one()
    rows = session.execute(
        stmt.order_by(models.Zone.name).limit(limit).offset(offset)
    ).all()
    return Page[Zone](
        items=[_zone_out(zone, key) for zone, key in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@zones_router.get("/{zone_id}", response_model=Zone)
def get_zone(
    zone_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("zones:read")),
):
    row, camera_key = _get_zone(session, zone_id)
    return _zone_out(row, camera_key)


@zones_router.patch("/{zone_id}", response_model=Zone)
def update_zone(
    zone_id: str,
    payload: ZoneUpdate,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("zones:manage")),
):
    row, camera_key = _get_zone(session, zone_id)
    data = payload.model_dump(exclude_unset=True)

    if "polygon" in data and data["polygon"] is not None:
        _require_simple_polygon(data["polygon"])
    if "name" in data and data["name"] != row.name:
        _check_unique_name(session, row.camera_id, data["name"], exclude_zone=zone_id)

    changed: list[str] = []
    for field, value in data.items():
        if field == "polygon":
            value = [list(p) for p in value]
            attr = "polygon"
        else:
            attr = "metadata_" if field == "metadata" else field
        if getattr(row, attr) != value:
            setattr(row, attr, value)
            changed.append(field)

    record_audit(
        session,
        principal,
        action="zone.update",
        resource_type="zone",
        resource_id=zone_id,
        details={"changed": changed, "camera_id": camera_key},
    )
    session.flush()
    return _zone_out(row, camera_key)


@zones_router.delete("/{zone_id}", status_code=204)
def delete_zone(
    zone_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("zones:manage")),
):
    row, camera_key = _get_zone(session, zone_id)
    record_audit(
        session,
        principal,
        action="zone.delete",
        resource_type="zone",
        resource_id=zone_id,
        details={"name": row.name, "camera_id": camera_key},
    )
    session.delete(row)
    session.flush()
