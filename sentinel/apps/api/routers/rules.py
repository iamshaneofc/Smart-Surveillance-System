from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from packages.db.audit import record_audit
from apps.api.deps import get_db_session, require_permission
from apps.api.errors import AppError
from packages.db import models
from packages.schemas.common import Page, Severity
from packages.schemas.rule import (
    IMPLEMENTED_RULE_TYPES,
    RuleCreate,
    RuleOut,
    RuleType,
    RuleUpdate,
    Schedule,
)

router = APIRouter(prefix="/rules", tags=["rules"])


def _require_supported_rule_type(rule_type: RuleType) -> None:
    if rule_type not in IMPLEMENTED_RULE_TYPES:
        raise AppError(
            "unsupported_rule_type",
            f"rule_type '{rule_type.value}' has no runtime implementation; "
            f"supported: {', '.join(sorted(t.value for t in IMPLEMENTED_RULE_TYPES))}",
            400,
        )


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


def _camera_key(session: Session, internal_id: str | None) -> str | None:
    if not internal_id:
        return None
    camera = session.get(models.Camera, internal_id)
    return camera.camera_id if camera else internal_id


def _rule_out(row: models.Rule, camera_key: str | None) -> RuleOut:
    return RuleOut(
        id=row.id,
        rule_id=row.rule_key,
        name=row.name,
        rule_type=row.rule_type,
        event_type=row.event_type,
        severity=row.severity,
        enabled=row.enabled,
        zone_ids=row.zone_ids or [],
        line=[tuple(p) for p in row.line] if row.line else None,
        params=row.params or {},
        schedule=Schedule(**(row.schedule or {})),
        cooldown_seconds=row.cooldown_seconds,
        confirm_seconds=row.confirm_seconds,
        min_confidence=row.min_confidence,
        camera_id=camera_key,
        site_id=row.site_id,
        version=row.version,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _get_rule(session: Session, rule_id: str) -> tuple[models.Rule, str | None]:
    row = session.execute(
        select(models.Rule).where(models.Rule.rule_key == rule_id)
    ).scalar_one_or_none()
    if row is None:
        raise AppError("not_found", f"rule '{rule_id}' not found", 404)
    return row, _camera_key(session, row.camera_id)


def _check_unique_rule_key(session: Session, rule_key: str, exclude_pk: str | None = None):
    stmt = select(models.Rule).where(models.Rule.rule_key == rule_key)
    if exclude_pk:
        stmt = stmt.where(models.Rule.id != exclude_pk)
    if session.execute(stmt).scalar_one_or_none() is not None:
        raise AppError("conflict", f"rule '{rule_key}' already exists", 409)


def _check_site(session: Session, site_id: str | None) -> None:
    if site_id and session.get(models.Site, site_id) is None:
        raise AppError("invalid_site", f"site '{site_id}' not found", 422)


def _bump_version(version: str) -> str:
    return str(int(version) + 1) if version.isdigit() else f"{version}+1"


@router.get("", response_model=Page[RuleOut])
def list_rules(
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("rules:read")),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    camera_id: str | None = None,
    enabled: bool | None = None,
    rule_type: str | None = None,
):
    stmt = select(models.Rule, models.Camera.camera_id).join(
        models.Camera, models.Rule.camera_id == models.Camera.id, isouter=True
    )
    if camera_id is not None:
        stmt = stmt.where(models.Camera.camera_id == camera_id)
    if enabled is not None:
        stmt = stmt.where(models.Rule.enabled == enabled)
    if rule_type is not None:
        stmt = stmt.where(models.Rule.rule_type == rule_type)
    total = session.execute(
        select(func.count()).select_from(stmt.subquery())
    ).scalar_one()
    rows = session.execute(
        stmt.order_by(models.Rule.rule_key).limit(limit).offset(offset)
    ).all()
    return Page[RuleOut](
        items=[_rule_out(rule, key) for rule, key in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@router.post("", status_code=201, response_model=RuleOut)
def create_rule(
    payload: RuleCreate,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("rules:manage")),
):
    _require_supported_rule_type(payload.rule_type)
    _check_unique_rule_key(session, payload.rule_id)
    _check_site(session, payload.site_id)

    camera_pk = None
    camera_key = None
    if payload.camera_id:
        camera = _get_camera(session, payload.camera_id)
        camera_pk = camera.id
        camera_key = camera.camera_id

    row = models.Rule(
        rule_key=payload.rule_id,
        camera_id=camera_pk,
        site_id=payload.site_id,
        name=payload.name,
        version="1",
        rule_type=payload.rule_type.value,
        event_type=payload.event_type,
        severity=payload.severity.value,
        enabled=payload.enabled,
        zone_ids=payload.zone_ids,
        line=[list(p) for p in payload.line] if payload.line else None,
        params=payload.params,
        schedule=payload.schedule.model_dump(),
        cooldown_seconds=payload.cooldown_seconds,
        confirm_seconds=payload.confirm_seconds,
        min_confidence=payload.min_confidence,
    )
    session.add(row)
    record_audit(
        session,
        principal,
        action="rule.create",
        resource_type="rule",
        resource_id=payload.rule_id,
        details={
            "rule_type": payload.rule_type.value,
            "camera_id": camera_key,
            "site_id": payload.site_id,
            "version": "1",
        },
    )
    session.flush()
    return _rule_out(row, camera_key)


@router.get("/{rule_id}", response_model=RuleOut)
def get_rule(
    rule_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("rules:read")),
):
    row, camera_key = _get_rule(session, rule_id)
    return _rule_out(row, camera_key)


@router.patch("/{rule_id}", response_model=RuleOut)
def update_rule(
    rule_id: str,
    payload: RuleUpdate,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("rules:manage")),
):
    row, camera_key = _get_rule(session, rule_id)
    data = payload.model_dump(exclude_unset=True)

    if "rule_id" in data and data["rule_id"] != row.rule_key:
        raise AppError(
            "rule_id_immutable",
            "rule_id is the event traceability key and cannot be renamed; create a new rule instead",
            409,
        )
    data.pop("rule_id", None)

    if "rule_type" in data and data["rule_type"] is not None:
        _require_supported_rule_type(RuleType(data["rule_type"]))
    if "site_id" in data:
        _check_site(session, data["site_id"])

    camera_key_changed = False
    if "camera_id" in data:
        previous_pk = row.camera_id
        if data["camera_id"]:
            camera = _get_camera(session, data["camera_id"])
            row.camera_id = camera.id
            camera_key = camera.camera_id
        else:
            row.camera_id = None
            camera_key = None
        camera_key_changed = row.camera_id != previous_pk
        data.pop("camera_id")

    changed: list[str] = ["camera_id"] if camera_key_changed else []
    for field, value in data.items():
        if field == "schedule":
            value = value.model_dump()
        elif field == "line" and value is not None:
            value = [list(p) for p in value]
        elif field in ("severity", "rule_type") and value is not None:
            value = value.value if hasattr(value, "value") else value
        if getattr(row, field) != value:
            setattr(row, field, value)
            changed.append(field)

    if changed:
        row.version = _bump_version(row.version)

    record_audit(
        session,
        principal,
        action="rule.update",
        resource_type="rule",
        resource_id=rule_id,
        details={"changed": changed, "version": row.version},
    )
    session.flush()
    return _rule_out(row, camera_key)


@router.delete("/{rule_id}", status_code=204)
def delete_rule(
    rule_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("rules:manage")),
):
    row, camera_key = _get_rule(session, rule_id)
    record_audit(
        session,
        principal,
        action="rule.delete",
        resource_type="rule",
        resource_id=rule_id,
        details={"name": row.name, "version": row.version, "camera_id": camera_key},
    )
    session.delete(row)
    session.flush()
