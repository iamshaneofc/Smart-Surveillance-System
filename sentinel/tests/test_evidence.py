from datetime import datetime, timedelta, timezone

from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus
from packages.schemas.evidence import EvidenceType
from services.camera.types import FramePacket
from services.evidence.buffer import RollingFrameBuffer
from services.evidence.retention import is_expired, resolve_expiry
from services.evidence.service import ClipWriter, EvidenceService
from services.evidence.store import MemoryEvidenceStore

TZ = timezone.utc


class FakeClipWriter(ClipWriter):
    def write(self, frames):
        return b"CLIPDATA", "video/mp4", "mp4"


def _packet(camera: str, ts: datetime, frame_id: int) -> FramePacket:
    return FramePacket(camera_id=camera, frame_id=frame_id, ts=ts, data=b"\xff\xd8FAKE", width=640, height=360)


def _event(ts: datetime, severity=Severity.CRITICAL) -> Event:
    now = ts
    return Event(
        event_id="evt_test1",
        camera_id="cam1",
        timestamp=ts,
        event_type="restricted_zone_intrusion",
        severity=severity,
        status=EventStatus.NEW,
        track_ids=[7],
        zone_id="z1",
        zone_name="Restricted",
        rule_id="r1",
        rule_name="Zone entry",
        conditions=[],
        model_versions={"detector": "stub:1"},
        created_at=now,
        updated_at=now,
    )


def test_buffer_retains_only_recent_window():
    buffer = RollingFrameBuffer("cam1", max_seconds=10.0, max_frames=100)
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    for i in range(11):
        buffer.append(_packet("cam1", t0 + timedelta(seconds=i), i))
    recent = buffer.recent(5.0, t0 + timedelta(seconds=10))
    assert len(recent) == 6
    assert recent[0].frame_id == 5


def test_buffer_frame_limit():
    buffer = RollingFrameBuffer("cam1", max_seconds=600.0, max_frames=5)
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    for i in range(10):
        buffer.append(_packet("cam1", t0 + timedelta(seconds=i), i))
    assert len(buffer) == 5


def test_evidence_capture_pre_and_post():
    store = MemoryEvidenceStore()
    buffer = RollingFrameBuffer("cam1", max_seconds=60.0)
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    for i in range(0, 11):
        buffer.append(_packet("cam1", t0 + timedelta(seconds=i - 10), i))

    service = EvidenceService(
        store=store,
        buffer=buffer,
        clip_writer=FakeClipWriter(),
        pre_seconds=5.0,
        post_seconds=3.0,
        snapshot_count=2,
    )
    event_time = t0
    event = _event(event_time)
    session = service.begin(event)
    assert session.pre_frames
    assert all(f.ts <= event_time for f in session.pre_frames)
    assert len(session.pre_frames) <= 6

    for i in range(1, 4):
        accepted = service.on_frame(_packet("cam1", event_time + timedelta(seconds=i), 100 + i))
        assert accepted is True

    assert service.on_frame(_packet("cam1", event_time + timedelta(seconds=10), 200)) is False
    assert service._sessions == {}
    assert store.exists("memory://cam1/evt_test1/clip.mp4")
    assert store.read("memory://cam1/evt_test1/clip.mp4") == b"CLIPDATA"


def test_evidence_finalize_produces_items_with_metadata_and_retention():
    store = MemoryEvidenceStore()
    buffer = RollingFrameBuffer("cam1", max_seconds=60.0)
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    for i in range(0, 11):
        buffer.append(_packet("cam1", t0 + timedelta(seconds=i - 5), i))

    service = EvidenceService(
        store=store,
        buffer=buffer,
        clip_writer=FakeClipWriter(),
        pre_seconds=5.0,
        post_seconds=0.0,
        snapshot_count=3,
        days_by_severity={"critical": 90, "high": 30, "medium": 14, "low": 3},
    )
    event = _event(t0, Severity.CRITICAL)
    service.begin(event)
    items = service.finalize("cam1")

    clips = [i for i in items if i.type == EvidenceType.CLIP]
    snapshots = [i for i in items if i.type == EvidenceType.SNAPSHOT]
    assert len(clips) == 1
    assert 1 <= len(snapshots) <= 3
    clip = clips[0]
    assert clip.sha256 is not None and len(clip.sha256) == 64
    assert clip.size_bytes == len(b"CLIPDATA")
    assert clip.metadata["event_type"] == "restricted_zone_intrusion"
    assert clip.metadata["model_versions"] == {"detector": "stub:1"}
    assert clip.expires_at is not None
    expected = resolve_expiry(Severity.CRITICAL, {"critical": 90}, t0)
    assert clip.expires_at == expected
    assert (expected - t0).days == 90


def test_retention_expiry_helpers():
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    assert resolve_expiry(Severity.LOW, {"low": 3}, t0) == t0 + timedelta(days=3)
    assert resolve_expiry(Severity.LOW, {"low": 0}, t0) is None
    assert resolve_expiry("missing", {}, t0) is None

    store = MemoryEvidenceStore()
    buffer = RollingFrameBuffer("cam1", max_seconds=60.0)
    buffer.append(_packet("cam1", t0 - timedelta(seconds=1), 1))
    buffer.append(_packet("cam1", t0 - timedelta(seconds=0), 2))
    service = EvidenceService(store=store, buffer=buffer, clip_writer=None, post_seconds=0.0, snapshot_count=1)
    event = _event(t0, Severity.LOW)
    service.begin(event)
    items = service.finalize("cam1")
    assert items
    assert is_expired(items[0], t0 + timedelta(days=4)) is True
    assert is_expired(items[0], t0 + timedelta(days=1)) is False
