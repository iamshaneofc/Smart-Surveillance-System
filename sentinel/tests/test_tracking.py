from datetime import datetime, timedelta, timezone

from services.inference.types import BBox, Detection
from services.tracking.iou import IOUTracker, NullTracker

TZ = timezone.utc


def _det(x: float, confidence: float = 0.9) -> Detection:
    return Detection(class_name="person", confidence=confidence, bbox=BBox(x, 0.4, 0.2, 0.4))


def test_iou_tracker_keeps_identity_across_frames():
    tracker = IOUTracker(iou_threshold=0.3, max_age=2.0, min_hits=2)
    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=TZ)

    frame1 = tracker.update([_det(0.1)], t0)
    assert frame1 == []

    frame2 = tracker.update([_det(0.12)], t0 + timedelta(seconds=0.5))
    assert len(frame2) == 1
    track_id = frame2[0].track_id
    assert frame2[0].state == "confirmed"

    frame3 = tracker.update([_det(0.14)], t0 + timedelta(seconds=1.0))
    assert len(frame3) == 1
    assert frame3[0].track_id == track_id
    assert frame3[0].hits == 3


def test_iou_tracker_marks_lost_then_drops():
    tracker = IOUTracker(iou_threshold=0.3, max_age=1.0, min_hits=1)
    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=TZ)
    tracker.update([_det(0.1)], t0)
    confirmed = tracker.update([_det(0.1)], t0 + timedelta(seconds=0.1))
    track_id = confirmed[0].track_id

    lost = tracker.update([], t0 + timedelta(seconds=0.5))
    assert len(lost) == 1
    assert lost[0].state == "lost"
    assert lost[0].track_id == track_id

    gone = tracker.update([], t0 + timedelta(seconds=2.0))
    assert gone == []


def test_iou_tracker_separate_objects_get_separate_ids():
    tracker = IOUTracker(min_hits=1)
    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=TZ)
    tracks = tracker.update([_det(0.1), _det(0.7)], t0)
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_iou_tracker_class_change_is_new_track():
    tracker = IOUTracker(min_hits=1, iou_threshold=0.3)
    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=TZ)
    first = tracker.update([_det(0.1)], t0)
    assert len(first) == 1
    vehicle = Detection(class_name="vehicle", confidence=0.8, bbox=BBox(0.1, 0.4, 0.2, 0.4))
    second = tracker.update([vehicle], t0 + timedelta(seconds=0.5))
    vehicle_tracks = [t for t in second if t.class_name == "vehicle"]
    assert len(vehicle_tracks) == 1
    assert vehicle_tracks[0].track_id != first[0].track_id
    assert any(t.class_name == "person" for t in second)


def test_null_tracker_assigns_fresh_ids():
    tracker = NullTracker()
    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=TZ)
    a = tracker.update([_det(0.1)], t0)
    b = tracker.update([_det(0.1)], t0 + timedelta(seconds=1))
    assert a[0].track_id != b[0].track_id
