from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from packages.schemas.common import HealthState, Severity

SUPPORTED_SOURCE_TYPES = ("file", "webcam", "rtsp", "synthetic")


class CameraSourceType(str, Enum):
    RTSP = "rtsp"
    ONVIF = "onvif"
    FILE = "file"
    WEBCAM = "webcam"
    SYNTHETIC = "synthetic"


class CameraState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    OFFLINE = "offline"
    STOPPED = "stopped"


class RetentionPolicy(BaseModel):
    pre_seconds: float = Field(default=10.0, ge=0, le=120)
    post_seconds: float = Field(default=10.0, ge=0, le=300)
    days_by_severity: dict[str, int] = Field(
        default_factory=lambda: {"critical": 90, "high": 30, "medium": 14, "low": 3}
    )


def _validate_source(source_type: CameraSourceType, stream_url: str) -> str:
    if source_type == CameraSourceType.ONVIF:
        raise ValueError(
            "source_type 'onvif' is not supported yet; "
            f"use one of: {', '.join(SUPPORTED_SOURCE_TYPES)}"
        )
    url = (stream_url or "").strip()
    if source_type == CameraSourceType.RTSP and not url.startswith(("rtsp://", "rtsps://")):
        raise ValueError("stream_url for source_type 'rtsp' must start with rtsp://")
    if source_type == CameraSourceType.FILE and not url:
        raise ValueError("stream_url is required for source_type 'file'")
    return url


class CameraBase(BaseModel):
    camera_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(min_length=1, max_length=128)
    location: str | None = None
    site_id: str | None = None
    source_type: CameraSourceType = CameraSourceType.RTSP
    enabled: bool = True
    detection_enabled: bool = True
    recording_enabled: bool = True
    detection_fps: float = Field(default=5.0, gt=0, le=60)
    width: int | None = Field(default=None, ge=64, le=7680)
    height: int | None = Field(default=None, ge=64, le=4320)
    timezone: str = "UTC"
    model_profile: str = "default"
    rule_profile: str = "default"
    retention: RetentionPolicy = Field(default_factory=RetentionPolicy)
    metadata: dict = Field(default_factory=dict)


class CameraCreate(CameraBase):
    stream_url: str = ""

    @model_validator(mode="after")
    def _check_source(self) -> "CameraCreate":
        _validate_source(self.source_type, self.stream_url)
        return self


class CameraUpdate(BaseModel):
    name: str | None = None
    location: str | None = None
    enabled: bool | None = None
    detection_enabled: bool | None = None
    recording_enabled: bool | None = None
    detection_fps: float | None = Field(default=None, gt=0, le=60)
    model_profile: str | None = None
    rule_profile: str | None = None
    stream_url: str | None = None
    retention: RetentionPolicy | None = None
    metadata: dict | None = None


class CameraOut(CameraBase):
    id: str
    stream_url_set: bool = True
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class CameraHealthSnapshot(BaseModel):
    camera_id: str
    state: CameraState
    health: HealthState
    fps: float = 0.0
    frame_drops: int = 0
    latency_ms: float | None = None
    reconnect_count: int = 0
    frames_processed: int = 0
    last_frame_at: datetime | None = None
    error: str | None = None
    ai_status: HealthState = HealthState.OFFLINE
    ts: datetime
    details: dict = Field(default_factory=dict)


class CameraHealthSummary(BaseModel):
    """Latest health per camera with a derived operational status.

    status: online | degraded | retrying | offline | disabled | unknown
    - offline   = camera stream down (state offline/stopped, health offline/error)
    - degraded  = camera streaming but detector/AI degraded
    - retrying  = connecting/reconnecting
    - disabled  = camera row disabled by configuration
    - unknown   = no health row yet (pipeline never ran for this camera)
    """

    camera_id: str
    name: str
    location: str | None = None
    site_id: str | None = None
    enabled: bool
    status: str
    stale: bool = False
    state: str | None = None
    health: str | None = None
    ai_status: str | None = None
    fps: float | None = None
    frame_drops: int | None = None
    reconnect_count: int | None = None
    last_frame_at: datetime | None = None
    error: str | None = None
    health_ts: datetime | None = None
