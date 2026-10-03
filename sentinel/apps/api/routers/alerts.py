from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from apps.api.deps import get_db_session, require_permission
from packages.db import models
from packages.schemas.alert import AlertOut
from packages.schemas.common import Page

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _alert_out(row: models.Alert) -> AlertOut:
    return AlertOut(
        id=row.id,
        event_id=row.event_id,
        channel=row.channel,
        status=row.status,
        target=row.target,
        attempts=row.attempts,
        error=row.error,
        sent_at=row.sent_at,
        created_at=row.created_at,
    )


@router.get("", response_model=Page[AlertOut])
def list_alerts(
    session: Session = Depends(get_db_session),
    principal=Depends(require_permission("alerts:read")),
    event_id: str | None = None,
    channel: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    stmt = select(models.Alert)
    if event_id is not None:
        stmt = stmt.where(models.Alert.event_id == event_id)
    if channel is not None:
        stmt = stmt.where(models.Alert.channel == channel)
    if status is not None:
        stmt = stmt.where(models.Alert.status == status)
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(
        stmt.order_by(models.Alert.created_at.desc(), models.Alert.id.desc())
        .limit(limit)
        .offset(offset)
    ).scalars()
    return Page[AlertOut](
        items=[_alert_out(r) for r in rows], total=int(total), limit=limit, offset=offset
    )
