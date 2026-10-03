from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from packages.schemas.common import Severity


class ZoneType(str, Enum):
    RESTRICTED = "restricted"
    HAZARDOUS = "hazardous"
    LOADING = "loading"
    ENTRANCE = "entrance"
    EMERGENCY = "emergency"
    PARKING = "parking"
    WAITING = "waiting"
    CUSTOM = "custom"


ZONE_ANCHORS = ("center", "top_center", "bottom_center")


def _check_polygon(value: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(value) < 3:
        raise ValueError("polygon requires at least 3 points")
    for x, y in value:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("polygon points must be normalized to [0,1]")
    return value


def _check_anchor(value: str) -> str:
    if value not in ZONE_ANCHORS:
        raise ValueError(f"anchor must be one of: {', '.join(ZONE_ANCHORS)}")
    return value


class Zone(BaseModel):
    id: str
    camera_id: str
    name: str = Field(min_length=1, max_length=128)
    zone_type: ZoneType = ZoneType.CUSTOM
    polygon: list[tuple[float, float]] = Field(min_length=3)
    anchor: str = "center"
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("polygon")
    @classmethod
    def polygon_valid(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return _check_polygon(value)

    @field_validator("anchor")
    @classmethod
    def anchor_valid(cls, value: str) -> str:
        return _check_anchor(value)


class ZoneCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    zone_type: ZoneType = ZoneType.CUSTOM
    polygon: list[tuple[float, float]] = Field(min_length=3)
    anchor: str = "center"
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("polygon")
    @classmethod
    def polygon_valid(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return _check_polygon(value)

    @field_validator("anchor")
    @classmethod
    def anchor_valid(cls, value: str) -> str:
        return _check_anchor(value)


class ZoneUpdate(BaseModel):
    name: str | None = None
    zone_type: ZoneType | None = None
    polygon: list[tuple[float, float]] | None = None
    anchor: str | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("polygon")
    @classmethod
    def polygon_valid(cls, value: list[tuple[float, float]] | None) -> list[tuple[float, float]] | None:
        if value is None:
            return value
        return _check_polygon(value)

    @field_validator("anchor")
    @classmethod
    def anchor_valid(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _check_anchor(value)


class ScheduleWindow(BaseModel):
    days: list[int] = Field(default_factory=lambda: list(range(7)))
    start: str = "00:00"
    end: str = "23:59"

    @field_validator("days")
    @classmethod
    def days_valid(cls, value: list[int]) -> list[int]:
        if any(d < 0 or d > 6 for d in value):
            raise ValueError("days must be 0 (Monday) .. 6 (Sunday)")
        return value

    @field_validator("start", "end")
    @classmethod
    def hhmm(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 2 or not (0 <= int(parts[0]) <= 23 and 0 <= int(parts[1]) <= 59):
            raise ValueError("time must be HH:MM")
        return value


class Schedule(BaseModel):
    """Empty windows list means the rule is always active."""

    windows: list[ScheduleWindow] = Field(default_factory=list)
    timezone: str = "UTC"


class RuleType(str, Enum):
    ZONE_ENTER = "zone_enter"
    ZONE_DWELL = "zone_dwell"
    LINE_CROSS = "line_cross"
    RESTRICTED_ZONE_INTRUSION = "restricted_zone_intrusion"
    OBJECT_COUNT = "object_count"
    PROXIMITY = "proximity"
    TEMPORAL_SIGNAL = "temporal_signal"


KNOWN_EVENT_TYPES = {
    "restricted_zone_intrusion",
    "hazardous_zone_intrusion",
    "zone_enter",
    "dwell_violation",
    "line_crossing",
    "after_hours_movement",
    "loitering",
    "abandoned_object",
    "crowd_anomaly",
    "weapon_candidate",
    "fall_candidate",
    "violence_candidate",
    "ppe_violation",
    "crowding",
}


class RuleDefinition(BaseModel):
    rule_id: str = Field(pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(min_length=1, max_length=128)
    rule_type: RuleType
    event_type: str = Field(min_length=1, max_length=64)
    severity: Severity = Severity.MEDIUM
    enabled: bool = True
    zone_ids: list[str] = Field(default_factory=list)
    line: list[tuple[float, float]] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    schedule: Schedule = Field(default_factory=Schedule)
    cooldown_seconds: float = Field(default=60.0, ge=0)
    confirm_seconds: float = Field(default=2.0, ge=0)
    min_confidence: float = Field(default=0.4, ge=0, le=1)


IMPLEMENTED_RULE_TYPES = {
    RuleType.ZONE_ENTER,
    RuleType.ZONE_DWELL,
    RuleType.LINE_CROSS,
    RuleType.RESTRICTED_ZONE_INTRUSION,
}


class RuleCreate(RuleDefinition):
    camera_id: str | None = None
    site_id: str | None = None


class RuleUpdate(BaseModel):
    rule_id: str | None = None
    name: str | None = None
    rule_type: RuleType | None = None
    event_type: str | None = None
    severity: Severity | None = None
    enabled: bool | None = None
    zone_ids: list[str] | None = None
    line: list[tuple[float, float]] | None = None
    params: dict[str, Any] | None = None
    schedule: Schedule | None = None
    cooldown_seconds: float | None = Field(default=None, ge=0)
    confirm_seconds: float | None = Field(default=None, ge=0)
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    camera_id: str | None = None
    site_id: str | None = None


class RuleOut(BaseModel):
    id: str
    rule_id: str
    name: str
    rule_type: RuleType
    event_type: str
    severity: Severity
    enabled: bool
    zone_ids: list[str]
    line: list[tuple[float, float]] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    schedule: Schedule = Field(default_factory=Schedule)
    cooldown_seconds: float
    confirm_seconds: float
    min_confidence: float
    camera_id: str | None = None
    site_id: str | None = None
    version: str
    created_at: datetime
    updated_at: datetime


class RulePack(BaseModel):
    name: str
    version: str = "0.1"
    industry: str
    description: str = ""
    default_classes: list[str] = Field(default_factory=lambda: ["person"])
    rules: list[RuleDefinition]
    created_at: datetime | None = None
