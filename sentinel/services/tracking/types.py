from dataclasses import dataclass, field
from datetime import datetime

from services.inference.types import BBox, Detection


@dataclass
class TrackState:
    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
    first_seen: datetime
    last_seen: datetime
    state: str = "tentative"
    hits: int = 1
    time_since_update: float = 0.0
    zone_ids: list[str] = field(default_factory=list)
    trajectory: list[tuple[datetime, float, float]] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        return max((self.last_seen - self.first_seen).total_seconds(), 0.0)


@dataclass
class DetectionFrame:
    ts: datetime
    detections: list[Detection]
