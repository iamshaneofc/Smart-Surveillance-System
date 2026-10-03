from datetime import datetime

from services.inference.types import Detection
from services.tracking.types import TrackState


class NullTracker:
    def __init__(self) -> None:
        self._next_id = 0

    def update(self, detections: list[Detection], ts: datetime) -> list[TrackState]:
        tracks: list[TrackState] = []
        for det in detections:
            track_id = self._next_id
            self._next_id += 1
            x, y = det.bbox.center()
            tracks.append(
                TrackState(
                    track_id=track_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    first_seen=ts,
                    last_seen=ts,
                    state="confirmed",
                    trajectory=[(ts, x, y)],
                )
            )
        return tracks

    def reset(self) -> None:
        self._next_id = 0


class IOUTracker:
    """Greedy IoU baseline - reference implementation, not the production tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_age: float = 1.0, min_hits: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id = 0
        self._active: list[TrackState] = []
        self._last_update: dict[int, datetime] = {}

    def update(self, detections: list[Detection], ts: datetime) -> list[TrackState]:
        unmatched_tracks = list(self._active)
        unmatched_dets = list(detections)
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(unmatched_tracks):
            for di, det in enumerate(unmatched_dets):
                if det.class_name != track.class_name:
                    continue
                iou = track.bbox.iou(det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True, key=lambda p: p[0])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = unmatched_tracks[ti]
            det = unmatched_dets[di]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.last_seen = ts
            track.hits += 1
            track.time_since_update = 0.0
            if track.state == "tentative" and track.hits >= self.min_hits:
                track.state = "confirmed"
            x, y = det.bbox.center()
            track.trajectory.append((ts, x, y))

        for ti, track in enumerate(unmatched_tracks):
            if ti in used_tracks:
                continue
            track.time_since_update = (ts - track.last_seen).total_seconds()
            if track.time_since_update > self.max_age:
                self._active.remove(track)
                continue
            if track.state == "confirmed":
                track.state = "lost"

        for di, det in enumerate(unmatched_dets):
            if di in used_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            x, y = det.bbox.center()
            track = TrackState(
                track_id=track_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                first_seen=ts,
                last_seen=ts,
                state="tentative",
                trajectory=[(ts, x, y)],
            )
            if self.min_hits <= 1:
                track.state = "confirmed"
            self._active.append(track)

        return [t for t in self._active if t.state in ("confirmed", "lost")]

    def reset(self) -> None:
        self._next_id = 0
        self._active = []
        self._last_update = {}
