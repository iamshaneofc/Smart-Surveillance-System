from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from packages.schemas.common import Severity


class EventStatus(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


ALLOWED_TRANSITIONS: dict[EventStatus, set[EventStatus]] = {
    EventStatus.NEW: {EventStatus.ACKNOWLEDGED, EventStatus.RESOLVED, EventStatus.DISMISSED},
    EventStatus.ACKNOWLEDGED: {EventStatus.RESOLVED, EventStatus.DISMISSED},
    EventStatus.RESOLVED: set(),
    EventStatus.DISMISSED: set(),
}


class ConditionEvidence(BaseModel):
    name: str
    operator: str
    actual: str | int | float | bool | None = None
    threshold: str | int | float | bool | None = None
    satisfied: bool


class EventDraft(BaseModel):
    camera_id: str
    timestamp: datetime
    event_type: str
    severity: Severity
    confidence: float | None = Field(default=None, ge=0, le=1)
    track_ids: list[int] = Field(default_factory=list)
    zone_id: str | None = None
    zone_name: str | None = None
    rule_id: str | None = None
    rule_name: str | None = None
    conditions: list[ConditionEvidence] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime
    event_type: str
    severity: Severity
    status: EventStatus = EventStatus.NEW
    confidence: float | None = Field(default=None, ge=0, le=1)
    summary: str = ""
    track_ids: list[int] = Field(default_factory=list)
    zone_id: str | None = None
    zone_name: str | None = None
    rule_id: str | None = None
    rule_name: str | None = None
    conditions: list[ConditionEvidence] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


class EventStatusUpdate(BaseModel):
    status: EventStatus
    note: str | None = None


class EventFilter(BaseModel):
    camera_id: str | None = None
    status: EventStatus | None = None
    severity: Severity | None = None
    event_type: str | None = None
    rule_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
