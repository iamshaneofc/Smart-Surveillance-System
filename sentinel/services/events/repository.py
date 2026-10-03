from __future__ import annotations

from typing import Protocol

from packages.schemas.event import Event, EventFilter


class EventRepository(Protocol):
    def save(self, event: Event) -> None: ...

    def get(self, event_id: str) -> Event | None: ...

    def list(self, filters: EventFilter) -> tuple[list[Event], int]: ...


class InMemoryEventRepository:
    def __init__(self) -> None:
        self._events: dict[str, Event] = {}

    def save(self, event: Event) -> None:
        self._events[event.event_id] = event.model_copy(deep=True)

    def get(self, event_id: str) -> Event | None:
        event = self._events.get(event_id)
        return event.model_copy(deep=True) if event else None

    def list(self, filters: EventFilter) -> tuple[list[Event], int]:
        items = [
            e.model_copy(deep=True)
            for e in self._all_filtered(filters)
        ]
        return items[filters.offset : filters.offset + filters.limit], len(items)

    def _all_filtered(self, filters: EventFilter) -> list[Event]:
        result = []
        for event in self._events.values():
            if filters.camera_id and event.camera_id != filters.camera_id:
                continue
            if filters.status and event.status != filters.status:
                continue
            if filters.severity and event.severity != filters.severity:
                continue
            if filters.event_type and event.event_type != filters.event_type:
                continue
            if filters.rule_id and event.rule_id != filters.rule_id:
                continue
            if filters.since and event.timestamp < filters.since:
                continue
            if filters.until and event.timestamp > filters.until:
                continue
            result.append(event)
        result.sort(key=lambda e: (e.timestamp, e.event_id), reverse=True)
        return result


def _row_to_event(row) -> Event:
    from packages.schemas.common import Severity

    return Event(
        event_id=row.id,
        camera_id=row.camera_id,
        timestamp=row.timestamp,
        event_type=row.event_type,
        severity=Severity(row.severity),
        status=row.status,
        confidence=row.confidence,
        summary=row.summary or "",
        track_ids=row.track_ids or [],
        zone_id=row.zone_id,
        zone_name=row.zone_name,
        rule_id=row.rule_id,
        rule_name=row.rule_name,
        conditions=row.conditions or [],
        model_versions=row.model_versions or {},
        evidence_ids=row.evidence_ids or [],
        metadata=row.metadata_ or {},
        created_at=row.created_at,
        updated_at=row.updated_at,
        acknowledged_by=row.acknowledged_by,
        acknowledged_at=row.acknowledged_at,
        resolved_at=row.resolved_at,
    )


class SqlEventRepository:
    def __init__(self, session) -> None:
        self._session = session

    def save(self, event: Event) -> None:
        from packages.db import models

        row = self._session.get(models.Event, event.event_id)
        if row is None:
            row = models.Event(id=event.event_id)
            self._session.add(row)
        row.camera_id = event.camera_id
        row.event_type = event.event_type
        row.severity = event.severity.value
        row.status = event.status.value
        row.timestamp = event.timestamp
        row.confidence = event.confidence
        row.summary = event.summary or ""
        row.track_ids = list(event.track_ids)
        row.zone_id = event.zone_id
        row.zone_name = event.zone_name
        row.rule_id = event.rule_id
        row.rule_name = event.rule_name
        row.dedup_key = (
            f"{event.camera_id}|{event.event_type}|{event.zone_id or '-'}|"
            + ",".join(str(t) for t in sorted(event.track_ids))
        )
        row.conditions = [c.model_dump() for c in event.conditions]
        row.model_versions = dict(event.model_versions)
        row.evidence_ids = list(event.evidence_ids)
        row.metadata_ = dict(event.metadata)
        row.acknowledged_by = event.acknowledged_by
        row.acknowledged_at = event.acknowledged_at
        row.resolved_at = event.resolved_at
        row.created_at = event.created_at
        row.updated_at = event.updated_at
        self._session.flush()

    def get(self, event_id: str) -> Event | None:
        from packages.db import models

        row = self._session.get(models.Event, event_id)
        return _row_to_event(row) if row else None

    def list(self, filters: EventFilter) -> tuple[list[Event], int]:
        from sqlalchemy import func, select

        from packages.db import models

        stmt = select(models.Event)
        if filters.camera_id:
            stmt = stmt.where(models.Event.camera_id == filters.camera_id)
        if filters.status:
            stmt = stmt.where(models.Event.status == filters.status.value)
        if filters.severity:
            stmt = stmt.where(models.Event.severity == filters.severity.value)
        if filters.event_type:
            stmt = stmt.where(models.Event.event_type == filters.event_type)
        if filters.rule_id:
            stmt = stmt.where(models.Event.rule_id == filters.rule_id)
        if filters.since:
            stmt = stmt.where(models.Event.timestamp >= filters.since)
        if filters.until:
            stmt = stmt.where(models.Event.timestamp <= filters.until)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = self._session.execute(count_stmt).scalar_one()

        rows = self._session.execute(
            stmt.order_by(models.Event.timestamp.desc(), models.Event.id.desc())
            .limit(filters.limit)
            .offset(filters.offset)
        ).scalars()
        return [_row_to_event(r) for r in rows], int(total)
