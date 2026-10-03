from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from services.tracking.types import TrackState


@dataclass
class ClipWindow:
    camera_id: str
    start: datetime
    end: datetime
    frame_refs: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return max((self.end - self.start).total_seconds(), 0.0)


@dataclass
class CandidateSignal:
    analyzer: str
    activity: str
    confidence: float
    ts: datetime
    window: ClipWindow
    track_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalContext:
    window: ClipWindow
    tracks: list[TrackState] = field(default_factory=list)
