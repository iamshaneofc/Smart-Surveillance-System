from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from packages.common.ids import new_id
from packages.common.timeutil import utcnow
from packages.db.base import Base


def _id() -> str:
    return new_id()


def _now() -> datetime:
    return utcnow()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Organization(Base, TimestampMixin):
    __tablename__ = "organization"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)


class Site(Base, TimestampMixin):
    __tablename__ = "site"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    org_id: Mapped[str] = mapped_column(ForeignKey("organization.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    location: Mapped[str | None] = mapped_column(String(255))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")


class Camera(Base, TimestampMixin):
    __tablename__ = "camera"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    camera_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    site_id: Mapped[str | None] = mapped_column(ForeignKey("site.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    location: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(16), default="rtsp")
    stream_url: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    detection_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    recording_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    detection_fps: Mapped[float] = mapped_column(Float, default=5.0)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    retention: Mapped[dict] = mapped_column(JSON, default=dict)
    model_profile: Mapped[str] = mapped_column(String(64), default="default")
    rule_profile: Mapped[str] = mapped_column(String(64), default="default")
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CameraHealth(Base):
    __tablename__ = "camera_health"
    __table_args__ = (Index("ix_camera_health_camera_ts", "camera_id", "ts"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    state: Mapped[str] = mapped_column(String(16))
    health: Mapped[str] = mapped_column(String(16))
    ai_status: Mapped[str] = mapped_column(String(16), default="offline")
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    frame_drops: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    reconnect_count: Mapped[int] = mapped_column(Integer, default=0)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    last_frame_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class Zone(Base, TimestampMixin):
    __tablename__ = "zone"
    __table_args__ = (Index("ix_zone_camera_name", "camera_id", "name", unique=True),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    zone_type: Mapped[str] = mapped_column(String(32), default="custom")
    polygon: Mapped[list] = mapped_column(JSON, default=list)
    anchor: Mapped[str] = mapped_column(String(32), default="center")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class RuleProfile(Base, TimestampMixin):
    __tablename__ = "rule_profile"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    industry: Mapped[str] = mapped_column(String(64), default="generic")
    description: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[str] = mapped_column(String(32), default="0.1")
    definition: Mapped[dict] = mapped_column(JSON, default=dict)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)


class Rule(Base, TimestampMixin):
    __tablename__ = "rule"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    rule_key: Mapped[str] = mapped_column(String(64), index=True)
    profile_id: Mapped[str | None] = mapped_column(ForeignKey("rule_profile.id", ondelete="CASCADE"))
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("camera.id", ondelete="CASCADE"), index=True)
    site_id: Mapped[str | None] = mapped_column(ForeignKey("site.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32), default="1")
    rule_type: Mapped[str] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16), default="medium")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    zone_ids: Mapped[list] = mapped_column(JSON, default=list)
    line: Mapped[list | None] = mapped_column(JSON)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    schedule: Mapped[dict] = mapped_column(JSON, default=dict)
    cooldown_seconds: Mapped[float] = mapped_column(Float, default=60.0)
    confirm_seconds: Mapped[float] = mapped_column(Float, default=2.0)
    min_confidence: Mapped[float] = mapped_column(Float, default=0.4)


class Event(Base, TimestampMixin):
    __tablename__ = "event"
    __table_args__ = (
        Index("ix_event_camera_timestamp", "camera_id", "timestamp"),
        Index("ix_event_status_severity", "status", "severity"),
        Index("ix_event_dedup", "dedup_key"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="new")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    summary: Mapped[str] = mapped_column(Text, default="")
    track_ids: Mapped[list] = mapped_column(JSON, default=list)
    zone_id: Mapped[str | None] = mapped_column(String(32))
    zone_name: Mapped[str | None] = mapped_column(String(128))
    rule_id: Mapped[str | None] = mapped_column(String(64))
    rule_name: Mapped[str | None] = mapped_column(String(128))
    dedup_key: Mapped[str] = mapped_column(String(255))
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    model_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_ids: Mapped[list] = mapped_column(JSON, default=list)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Track(Base):
    __tablename__ = "track"
    __table_args__ = (Index("ix_track_camera_track", "camera_id", "external_track_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    external_track_id: Mapped[int] = mapped_column(Integer)
    event_id: Mapped[str | None] = mapped_column(ForeignKey("event.id", ondelete="SET NULL"), index=True)
    class_name: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    bbox: Mapped[list | dict] = mapped_column(JSON, default=dict)
    trajectory: Mapped[list] = mapped_column(JSON, default=list)
    current_zone: Mapped[str | None] = mapped_column(String(128))
    previous_zones: Mapped[list] = mapped_column(JSON, default=list)
    dwell_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class Evidence(Base, TimestampMixin):
    __tablename__ = "evidence"
    __table_args__ = (Index("ix_evidence_expires", "expires_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("event.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(16))
    uri: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    content_type: Mapped[str | None] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    storage_backend: Mapped[str] = mapped_column(String(32), default="local")
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class Alert(Base, TimestampMixin):
    __tablename__ = "alert"
    __table_args__ = (Index("ix_alert_event_channel", "event_id", "channel"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("event.id", ondelete="CASCADE"), index=True)
    channel: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    target: Mapped[str | None] = mapped_column(String(255))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Role(Base, TimestampMixin):
    __tablename__ = "role"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    permissions: Mapped[list] = mapped_column(JSON, default=list)


class User(Base, TimestampMixin):
    __tablename__ = "user"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    org_id: Mapped[str | None] = mapped_column(ForeignKey("organization.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    hashed_password: Mapped[str | None] = mapped_column(String(255))
    api_key_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    role_id: Mapped[str | None] = mapped_column(ForeignKey("role.id"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_ts", "ts"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    org_id: Mapped[str | None] = mapped_column(String(32), index=True)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    ip: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Model(Base, TimestampMixin):
    __tablename__ = "model"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    task: Mapped[str] = mapped_column(String(64), default="detection")
    family: Mapped[str | None] = mapped_column(String(64))
    source: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(128))
    commercial_status: Mapped[str] = mapped_column(String(32), default="research-only")
    notes: Mapped[str] = mapped_column(Text, default="")


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_version"
    __table_args__ = (Index("ix_model_version_unique", "model_id", "version", unique=True),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    model_id: Mapped[str] = mapped_column(ForeignKey("model.id", ondelete="CASCADE"), index=True)
    version: Mapped[str] = mapped_column(String(64))
    uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    training_data: Mapped[dict] = mapped_column(JSON, default=dict)
    license: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="candidate")


class Deployment(Base, TimestampMixin):
    __tablename__ = "deployment"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_version.id"), index=True)
    environment: Mapped[str] = mapped_column(String(64), default="dev")
    target: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="active")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
