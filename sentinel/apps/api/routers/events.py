from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.deps import get_bus, get_current_principal, get_db_session, require_permission
from apps.api.errors import AppError
from packages.db import models
from packages.schemas.auth import Principal
from packages.schemas.common import Page, Severity
from packages.schemas.event import Event, EventFilter, EventStatus, EventStatusUpdate
from services.events.repository import SqlEventRepository

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=Page[Event])
def list_events(
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("events:read")),
    camera_id: str | None = None,
    status: EventStatus | None = None,
    severity: Severity | None = None,
    event_type: str | None = None,
    rule_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    repository = SqlEventRepository(session)
    filters = EventFilter(
        camera_id=camera_id,
        status=status,
        severity=severity,
        event_type=event_type,
        rule_id=rule_id,
        since=since if since is not None else start_time,
        until=until if until is not None else end_time,
        limit=limit,
        offset=offset,
    )
    items, total = repository.list(filters)
    return Page[Event](items=items, total=total, limit=limit, offset=offset)


@router.get("/{event_id}", response_model=Event)
def get_event(
    event_id: str,
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("events:read")),
):
    repository = SqlEventRepository(session)
    event = repository.get(event_id)
    if event is None:
        raise AppError("not_found", f"event '{event_id}' not found", 404)
    return event


@router.post("/{event_id}/status", response_model=Event)
def update_event_status(
    event_id: str,
    payload: EventStatusUpdate,
    session: Session = Depends(get_db_session),
    principal: Principal = Depends(get_current_principal),
    bus=Depends(get_bus),
):
    permission_map = {
        EventStatus.ACKNOWLEDGED: "events:ack",
        EventStatus.RESOLVED: "events:ack",
        EventStatus.DISMISSED: "events:dismiss",
    }
    required = permission_map.get(payload.status, "events:ack")
    if not principal.has(required):
        raise AppError("forbidden", f"missing permission: {required}", 403)

    repository = SqlEventRepository(session)
    event = repository.get(event_id)
    if event is None:
        raise AppError("not_found", f"event '{event_id}' not found", 404)

    from services.events.engine import EventEngine

    engine = EventEngine()
    try:
        engine.transition(event, payload.status, actor=principal.user)
    except ValueError as exc:
        raise AppError("invalid_transition", str(exc), 409) from exc
    repository.save(event)

    session.add(
        models.AuditLog(
            actor=principal.user,
            action=f"event.{payload.status.value}",
            resource_type="event",
            resource_id=event_id,
            details={"note": payload.note} if payload.note else {},
        )
    )
    session.commit()

    try:
        bus.publish(
            "events.updated",
            {
                "event_id": event.event_id,
                "camera_id": event.camera_id,
                "status": event.status.value,
                "actor": principal.user,
            },
        )
    except Exception:
        pass
    return event
