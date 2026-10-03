"""F1 vertical slice: frames -> detection -> tracking -> rule -> event -> evidence."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from packages.schemas.common import HealthState, Severity
from packages.schemas.event import EventStatus
from packages.schemas.rule import RuleDefinition, RuleType
from services.camera.types import FramePacket
from services.events.engine import EventEngine
from services.evidence.buffer import RollingFrameBuffer
from services.evidence.service import EvidenceService, OpenCvClipWriter
from services.evidence.store import MemoryEvidenceStore
from services.inference.hog import DetectorError
from services.inference.types import BBox, Detection, ModelInfo
from services.pipeline.pipeline import SurveillancePipeline
from services.rules.base import RuleSet
from services.rules.context import SpatioTemporalState, ZoneContext
from services.tracking.iou import IOUTracker

T0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
STEP = timedelta(seconds=0.2)
CAMERA = "cam1"

ZONE = ZoneContext(
    id="restricted",
    name="Restricted Area",
    zone_type="restricted",
    polygon=[(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)],
)
# IoU between these two boxes is 0.5, so one track survives zone crossings.
OUTSIDE = BBox(0.22, 0.4, 0.12, 0.3)  # center x=0.28 -> outside
INSIDE = BBox(0.26, 0.4, 0.12, 0.3)  # center x=0.32 -> inside

try:
    import cv2
    import numpy as np

    _ok, _buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    JPG = _buf.tobytes() if _ok else b"\xff\xd8"
except Exception:  # pragma: no cover - opencv optional for most tests
    JPG = b"\xff\xd8"
HAS_CV2 = JPG.startswith(b"\xff\xd8")


class ScriptedDetector:
    info = ModelInfo(
        name="scripted-detector",
        version="1.0",
        family="test",
        license="internal",
        classes=("person",),
    )

    def __init__(self, where):
        self._where = where

    def detect(self, frame: FramePacket) -> list[Detection]:
        bbox = self._where(frame.frame_id)
        if bbox is None:
            return []
        return [
            Detection(
                class_name="person",
                confidence=0.9,
                bbox=bbox,
                timestamp=frame.ts,
                model_id=self.info.name,
                model_version=self.info.version,
            )
        ]

    def warmup(self):
        return None

    def close(self):
        return None


class BrokenDetector(ScriptedDetector):
    def detect(self, frame: FramePacket) -> list[Detection]:
        raise DetectorError("model exploded")


class BrokenTracker:
    def update(self, detections, ts):
        raise RuntimeError("tracker exploded")

    def reset(self):
        return None


class RecordingBus:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))


def frame(i: int, data: bytes = b"frame") -> FramePacket:
    return FramePacket(
        camera_id=CAMERA,
        frame_id=i,
        ts=T0 + STEP * i,
        data=data,
        width=64,
        height=48,
    )


def build(where, **overrides):
    definition = RuleDefinition(
        rule_id="restricted-zone-entry",
        name="Restricted zone intrusion",
        rule_type=RuleType.RESTRICTED_ZONE_INTRUSION,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        zone_ids=["restricted"],
        params={"classes": ["person"]},
        min_confidence=overrides.pop("min_confidence", 0.5),
        confirm_seconds=overrides.pop("confirm", 2.0),
        cooldown_seconds=overrides.pop("cooldown", 5.0),
    )
    detector = overrides.pop("detector", None) or ScriptedDetector(where)
    tracker = overrides.pop("tracker", None) or IOUTracker(min_hits=1)
    post_seconds = overrides.pop("post_seconds", 1.0)
    pre_seconds = overrides.pop("pre_seconds", 2.0)
    clip = overrides.pop("clip", HAS_CV2)
    on_finalize = overrides.pop("on_finalize", None)
    event_sink = overrides.pop("event_sink", None)
    bus = overrides.pop("bus", None)
    on_ai_status = overrides.pop("on_ai_status", None)

    engine = EventEngine(rules={definition.rule_id: definition})
    buffer = RollingFrameBuffer(CAMERA, max_seconds=30.0)
    store = MemoryEvidenceStore()
    evidence = EvidenceService(
        store=store,
        buffer=buffer,
        clip_writer=OpenCvClipWriter() if clip else None,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        snapshot_count=3,
        on_finalize=on_finalize,
    )
    pipeline = SurveillancePipeline(
        camera_id=CAMERA,
        detector=detector,
        tracker=tracker,
        rule_set=RuleSet([definition]),
        engine=engine,
        spatial=SpatioTemporalState(),
        zones=[ZONE],
        evidence_service=evidence,
        rule_version="0.1",
        event_sink=event_sink,
        buffer=buffer,
        bus=bus,
        on_ai_status=on_ai_status,
    )
    return pipeline, engine, store, evidence


def feed(pipeline, frames):
    events = []
    for packet in frames:
        events.extend(pipeline.process_frame(packet))
    return events


def scripted_where(*ranges):
    """Return a detector script: person inside the zone for (start, end) frame ranges."""

    def where(i: int):
        for start, end in ranges:
            if i >= start and (end is None or i <= end):
                return INSIDE
        return OUTSIDE

    return where


def test_event_created_after_confirmation_window():
    pipeline, engine, store, evidence = build(scripted_where((5, None)))
    events = feed(pipeline, [frame(i, JPG if HAS_CV2 else b"f") for i in range(21)])

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "restricted_zone_intrusion"
    assert event.severity == Severity.HIGH
    assert event.status == EventStatus.NEW
    assert event.timestamp == T0 + STEP * 14  # pending started frame 5, confirmed frame 15
    assert event.confidence == pytest.approx(0.9)
    assert event.zone_id == "restricted"
    assert event.zone_name == "Restricted Area"
    assert event.rule_id == "restricted-zone-entry"
    assert event.metadata["rule_version"] == "0.1"
    assert "Restricted Area" in event.summary
    assert "1.8 seconds" in event.summary
    assert pipeline.events_created == 1
    assert engine.open_events() == [event]
    # nothing before confirmation
    assert pipeline.frames_processed == 21


def test_transient_intrusion_never_confirms():
    # inside frames 5..9 (0.8s < 2s confirm), then leaves: pending must be cancelled
    pipeline, engine, store, evidence = build(scripted_where((5, 9)))
    events = feed(pipeline, [frame(i) for i in range(40)])
    assert events == []
    assert engine.open_events() == []
    assert engine.pending_keys() == set()
    assert pipeline.events_created == 0


def test_long_intrusion_creates_exactly_one_event_for_10_seconds():
    pipeline, engine, store, evidence = build(scripted_where((5, None)))
    events = feed(pipeline, [frame(i) for i in range(60)])  # inside 5..59 => >10s
    assert len(events) == 1
    assert pipeline.events_created == 1
    assert len(engine.open_events()) == 1


def test_event_resolves_on_leave_and_cooldown_blocks_then_allows_reentry():
    # frames: 0-4 out, 5-30 in (event@15), 31-35 out (resolved@31),
    # 36+ in again -> cooldown from frame 31 blocks until gap>=5s (frame 56),
    # confirmation window then confirms at frame 66.
    pipeline, engine, store, evidence = build(scripted_where((5, 30), (36, None)), cooldown=5.0)
    per_frame = {}
    for i in range(70):
        produced = pipeline.process_frame(frame(i))
        if produced:
            per_frame[i] = produced

    assert sorted(per_frame) == [15, 66]
    first, second = per_frame[15][0], per_frame[66][0]
    assert first.event_id != second.event_id
    assert first.status == EventStatus.RESOLVED
    assert first.resolved_at == T0 + STEP * 31
    assert second.status == EventStatus.NEW
    assert second.severity == Severity.HIGH  # escalation requires 3rd recurrence
    assert second.timestamp >= first.resolved_at + timedelta(seconds=5.0)


def test_detector_failure_survives_and_reports_degraded():
    ai: list[HealthState] = []
    pipeline, engine, store, evidence = build(
        lambda i: INSIDE, detector=BrokenDetector(lambda i: INSIDE), on_ai_status=ai.append
    )
    events = feed(pipeline, [frame(i) for i in range(5)])
    assert events == []
    assert pipeline.detector_errors == 5
    assert pipeline.frames_processed == 5
    assert ai and ai[-1] == HealthState.DEGRADED
    assert HealthState.HEALTHY not in ai


def test_tracker_failure_survives():
    pipeline, engine, store, evidence = build(lambda i: INSIDE, tracker=BrokenTracker())
    events = pipeline.process_frame(frame(0))
    assert events == []
    assert pipeline.tracker_errors == 1


def test_evidence_created_only_for_event_with_hashes_and_links():
    finalized = {}

    def on_finalize(event, items):
        finalized["event"] = event
        finalized["items"] = items
        event.evidence_ids = [item.evidence_id for item in items]

    pipeline, engine, store, evidence = build(
        scripted_where((0, None)), on_finalize=on_finalize, post_seconds=1.0, pre_seconds=2.0
    )
    events = feed(pipeline, [frame(i, JPG) for i in range(20)])
    assert len(events) == 1
    event = events[0]

    # deadline = event.ts (frame 9, T0+1.8) + 1.0s -> finalized by frame 15 (T0+3.0)
    assert "items" in finalized
    items = finalized["items"]
    assert finalized["event"].event_id == event.event_id
    assert event.evidence_ids == [item.evidence_id for item in items]
    assert all(item.event_id == event.event_id for item in items)
    assert all(item.camera_id == CAMERA for item in items)

    types = {item.type.value for item in items}
    assert "snapshot" in types
    if HAS_CV2:
        assert "clip" in types
    for item in items:
        assert item.sha256 == hashlib.sha256(store.read(item.uri)).hexdigest()
        assert item.size_bytes == len(store.read(item.uri))
        assert item.metadata["rule_version"] == "0.1"
        assert item.metadata["rule_id"] == "restricted-zone-entry"
        assert item.expires_at is not None

    snapshots = [i for i in items if i.type.value == "snapshot"]
    assert snapshots
    assert all(s.content_type == "image/jpeg" for s in snapshots)


def test_normal_activity_creates_no_events_or_evidence():
    pipeline, engine, store, evidence = build(lambda i: OUTSIDE)
    events = feed(pipeline, [frame(i) for i in range(50)])
    assert events == []
    assert engine.open_events() == []
    assert engine.pending_keys() == set()
    assert store._blobs == {}
    assert evidence._sessions == {}
    assert pipeline.events_created == 0


def test_event_persisted_and_published():
    sink_events = []
    bus = RecordingBus()
    pipeline, engine, store, evidence = build(
        scripted_where((5, None)), event_sink=sink_events.append, bus=bus
    )
    events = feed(pipeline, [frame(i) for i in range(40)])

    assert len(events) == 1
    assert [e.event_id for e in sink_events] == [events[0].event_id]
    topics = [t for t, _ in bus.published]
    assert topics == ["events.created"]
    payload = bus.published[0][1]
    assert payload["event_id"] == events[0].event_id
    assert payload["event_type"] == "restricted_zone_intrusion"


def test_event_sink_called_again_when_auto_resolved():
    sink_events = []
    pipeline, engine, store, evidence = build(
        scripted_where((5, 30)), event_sink=lambda e: sink_events.append(e.model_copy(deep=True))
    )
    feed(pipeline, [frame(i) for i in range(40)])
    assert len(sink_events) == 2
    assert sink_events[0].status == EventStatus.NEW
    assert sink_events[1].status == EventStatus.RESOLVED
    assert sink_events[0].event_id == sink_events[1].event_id


def test_healthy_detector_reports_healthy_ai_status():
    ai: list[HealthState] = []
    pipeline, engine, store, evidence = build(lambda i: OUTSIDE, on_ai_status=ai.append)
    feed(pipeline, [frame(i) for i in range(3)])
    assert ai and set(ai) == {HealthState.HEALTHY}


