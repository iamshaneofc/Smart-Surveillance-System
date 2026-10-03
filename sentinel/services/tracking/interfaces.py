from datetime import datetime
from typing import Protocol

from services.inference.types import Detection
from services.tracking.types import TrackState


class Tracker(Protocol):
    def update(self, detections: list[Detection], ts: datetime) -> list[TrackState]: ...

    def reset(self) -> None: ...
