from dataclasses import dataclass, field
from datetime import datetime

from services.rules.geometry import anchor_point, bbox_center, point_in_polygon
from services.tracking.types import TrackState


@dataclass
class ZoneContext:
    id: str
    name: str
    zone_type: str
    polygon: list[tuple[float, float]]
    enabled: bool = True
    anchor: str = "center"


@dataclass
class ZoneTransition:
    ts: datetime
    track_id: int
    zone_id: str
    zone_name: str
    kind: str  # enter | exit
    class_name: str
    confidence: float


@dataclass
class EnrichedTrack:
    track: TrackState
    center: tuple[float, float]
    zone_ids: list[str] = field(default_factory=list)


@dataclass
class TrackSpatialState:
    zone_since: dict[str, datetime] = field(default_factory=dict)
    previous_zone_ids: list[str] = field(default_factory=list)
    previous_center: tuple[float, float] | None = None
    last_center: tuple[float, float] | None = None


@dataclass
class EvaluationContext:
    camera_id: str
    camera_timezone: str
    ts: datetime
    zones: list[ZoneContext]
    tracks: list[EnrichedTrack]
    transitions: list[ZoneTransition] = field(default_factory=list)
    spatial: "SpatioTemporalState | None" = None
    metadata: dict = field(default_factory=dict)


class SpatioTemporalState:
    def __init__(self) -> None:
        self._tracks: dict[int, TrackSpatialState] = {}
        self.transitions: list[ZoneTransition] = []

    def enrich(self, tracks: list[TrackState], zones: list[ZoneContext]) -> list[EnrichedTrack]:
        enriched: list[EnrichedTrack] = []
        for track in tracks:
            zone_ids = []
            for z in zones:
                if not z.enabled:
                    continue
                point = anchor_point(
                    track.bbox.x, track.bbox.y, track.bbox.w, track.bbox.h, z.anchor
                )
                if point_in_polygon(point, z.polygon):
                    zone_ids.append(z.id)
            center = bbox_center(track.bbox.x, track.bbox.y, track.bbox.w, track.bbox.h)
            track.zone_ids = list(zone_ids)
            enriched.append(EnrichedTrack(track=track, center=center, zone_ids=zone_ids))
        return enriched

    def update(
        self,
        tracks: list[EnrichedTrack],
        zones: list[ZoneContext],
        ts: datetime,
    ) -> list[ZoneTransition]:
        zones_by_id = {z.id: z for z in zones}
        fresh: list[ZoneTransition] = []
        seen: set[int] = set()
        for item in tracks:
            track = item.track
            seen.add(track.track_id)
            state = self._tracks.setdefault(track.track_id, TrackSpatialState())
            current = set(item.zone_ids)
            previous = set(state.zone_since)
            for zone_id in current - previous:
                state.zone_since[zone_id] = ts
                zone = zones_by_id.get(zone_id)
                fresh.append(
                    ZoneTransition(
                        ts=ts,
                        track_id=track.track_id,
                        zone_id=zone_id,
                        zone_name=zone.name if zone else zone_id,
                        kind="enter",
                        class_name=track.class_name,
                        confidence=track.confidence,
                    )
                )
            for zone_id in previous - current:
                state.zone_since.pop(zone_id, None)
                zone = zones_by_id.get(zone_id)
                fresh.append(
                    ZoneTransition(
                        ts=ts,
                        track_id=track.track_id,
                        zone_id=zone_id,
                        zone_name=zone.name if zone else zone_id,
                        kind="exit",
                        class_name=track.class_name,
                        confidence=track.confidence,
                    )
                )
            state.previous_zone_ids = list(current)
            state.previous_center = state.last_center
            state.last_center = item.center
        for track_id in list(self._tracks):
            if track_id not in seen:
                del self._tracks[track_id]
        self.transitions = fresh
        return fresh

    def dwell_seconds(self, track_id: int, zone_id: str, ts: datetime) -> float | None:
        state = self._tracks.get(track_id)
        if state is None or zone_id not in state.zone_since:
            return None
        return max((ts - state.zone_since[zone_id]).total_seconds(), 0.0)

    def zone_since(self, track_id: int, zone_id: str) -> datetime | None:
        state = self._tracks.get(track_id)
        if state is None:
            return None
        return state.zone_since.get(zone_id)

    def last_center(self, track_id: int) -> tuple[float, float] | None:
        state = self._tracks.get(track_id)
        return state.last_center if state else None

    def previous_center(self, track_id: int) -> tuple[float, float] | None:
        state = self._tracks.get(track_id)
        return state.previous_center if state else None
