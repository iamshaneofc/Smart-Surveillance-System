from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiKeySpec(BaseModel):
    key: str
    user: str
    roles: list[str] = Field(default_factory=list)


class EvidenceSettings(BaseModel):
    pre_seconds: float = 10.0
    post_seconds: float = 10.0
    root: str = "./var/evidence"
    retention_days_by_severity: dict[str, int] = Field(
        default_factory=lambda: {"critical": 90, "high": 30, "medium": 14, "low": 3}
    )
    snapshot_count: int = 3
    allow_active_event_deletion: bool = False


class AlertSettings(BaseModel):
    """Alert delivery configuration (environment: SENTINEL_ALERTS__*)."""

    enabled: bool = False
    channels: list[str] = Field(default_factory=lambda: ["in_app"])
    webhook_url: str = ""
    webhook_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    webhook_max_retries: int = Field(default=2, ge=0, le=5)
    webhook_backoff_seconds: float = Field(default=0.5, ge=0, le=30)
    cooldown_seconds: float = Field(default=120.0, ge=0)
    rate_cap_per_minute: int = Field(default=30, ge=1)
    queue_max: int = Field(default=100, ge=1, le=10000)


class ZoneSpec(BaseModel):
    id: str = "restricted"
    name: str = "Restricted Area"
    zone_type: str = "restricted"
    polygon: list[tuple[float, float]] = Field(min_length=3)
    anchor: str = "center"
    enabled: bool = True


class PipelineSettings(BaseModel):
    """F1 vertical-slice pipeline configuration (environment: SENTINEL_PIPELINE__*)."""

    enabled: bool = False
    camera_id: str = "cam-01"
    camera_name: str = "Demo camera"
    camera_location: str = ""
    timezone: str = "UTC"
    source_type: str = "file"  # file | rtsp | webcam | synthetic
    stream_url: str = ""  # file path or rtsp:// url (never logged)
    detection_fps: float = 5.0
    detector_profile: str = "hog"  # hog | stub | null | yolox
    detector_options: dict = Field(default_factory=dict)
    tracker: str = "iou"  # iou (DEVELOPMENT TRACKER) | null
    rule_pack: str = "factory.yaml"
    zones: list[ZoneSpec] = Field(default_factory=list)
    health_flush_seconds: float = 5.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "SENTINEL"
    env: str = "dev"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_json: bool = False

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    database_url: str = "sqlite:///./var/sentinel.db"
    bus_url: str = "memory://"

    auth_mode: Literal["disabled", "api_key"] = "disabled"
    auth_api_keys: list[ApiKeySpec] = Field(default_factory=list)
    rate_limit_per_minute: int = 120

    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)

    camera_default_detection_fps: float = 5.0
    camera_reconnect_initial_seconds: float = 1.0
    camera_reconnect_max_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
