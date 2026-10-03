from dataclasses import dataclass, field
from datetime import datetime

from packages.schemas.camera import CameraState


class SourceError(Exception):
    pass


class SourceUnavailableError(SourceError):
    pass


class SourceReadError(SourceError):
    pass


@dataclass(slots=True)
class FramePacket:
    camera_id: str
    frame_id: int
    ts: datetime
    data: bytes
    width: int = 0
    height: int = 0
    stream: str = "detect"


@dataclass
class SourceStats:
    frames_read: int = 0
    opened_at: datetime | None = None
    last_frame_at: datetime | None = None
    errors: list[str] = field(default_factory=list)
