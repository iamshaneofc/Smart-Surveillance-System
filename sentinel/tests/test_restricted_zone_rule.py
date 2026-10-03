from datetime import datetime, timedelta, timezone

from packages.schemas.common import Severity
from packages.schemas.rule import RuleDefinition, RuleType, Schedule, ScheduleWindow
from services.inference.types import BBox
from services.rules.builtin import RestrictedZoneIntrusionRule
from services.rules.context import EvaluationContext, SpatioTemporalState, ZoneContext
from services.rules.geometry import ANCHOR_POINTS, anchor_point, bbox_center
from services.tracking.types import TrackState

T0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
STEP = timedelta(seconds=0.5)

VERTICAL_ZONE = ZoneContext(
    id="restricted",
    name="Restricted Area",
    zone_type="restricted",
    polygon=[(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)],
)
HORIZONTAL_ZONE = ZoneContext(
    id="line-zone",
    name="Line zone",
    zone_type="restricted",
    polygon=[(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)],
)
BOTTOM_ANCHOR_ZONE = ZoneContext(
    id="bottom",
    name="Bottom anchored",
    zone_type="restricted",
    polygon=[(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)],
    anchor="bottom_center",
)


def make_rule(**overrides) -> RestrictedZoneIntrusionRule:
    kwargs = dict(
        rule_id="restricted-zone-entry",
        name="Restricted zone intrusion",
        rule_type=RuleType.RESTRICTED_ZONE_INTRUSION,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        zone_ids=["restricted"],
        params={"classes": ["person"]},
        min_confidence=0.5,
        confirm_seconds=2.0,
        cooldown_seconds=30.0,
    )
    kwargs.update(overrides)
    return RestrictedZoneIntrusionRule(RuleDefinition(**kwargs))


def make_track(track_id, bbox, ts, confidence=0.9, class_name="person"):
    return TrackState(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        first_seen=ts,
        last_seen=ts,
        state="confirmed",
    )


def evaluate(rule, spatial, zones, tracks, ts):
    enriched = spatial.enrich(tracks, zones)
    spatial.update(enriched, zones, ts)
    ctx = EvaluationContext(
        camera_id="cam1",
        camera_timezone="UTC",
        ts=ts,
        zones=zones,
        tracks=enriched,
        transitions=spatial.transitions,
        spatial=spatial,
    )
    return rule.evaluate(ctx)


def test_anchor_points():
    import pytest

    assert ANCHOR_POINTS == ("center", "top_center", "bottom_center")
    assert anchor_point(0.2, 0.2, 0.2, 0.4) == pytest.approx((0.3, 0.4))
    assert anchor_point(0.2, 0.2, 0.2, 0.4, "top_center") == pytest.approx((0.3, 0.2))
    assert anchor_point(0.2, 0.2, 0.2, 0.4, "bottom_center") == pytest.approx((0.3, 0.6))
    assert anchor_point(0.2, 0.2, 0.2, 0.4, "unknown") == pytest.approx((0.3, 0.4))


def test_enrichment_uses_zone_anchor_and_keeps_bbox_center():
    bbox = BBox(0.2, 0.3, 0.2, 0.3)  # center y=0.45 (above line), bottom y=0.6 (below line)
    track = make_track(1, bbox, T0)
    spatial = SpatioTemporalState()

    center_enriched = spatial.enrich([track], [HORIZONTAL_ZONE])
    assert center_enriched[0].zone_ids == []
    assert track.zone_ids == []

    bottom_enriched = spatial.enrich([track], [BOTTOM_ANCHOR_ZONE])
    assert bottom_enriched[0].zone_ids == ["bottom"]
    assert track.zone_ids == ["bottom"]
    assert bottom_enriched[0].center == bbox_center(0.2, 0.3, 0.2, 0.3)


def test_rule_emits_every_frame_while_inside():
    rule = make_rule()
    spatial = SpatioTemporalState()
    inside = BBox(0.45, 0.4, 0.1, 0.3)

    match1 = evaluate(rule, spatial, [VERTICAL_ZONE], [make_track(1, inside, T0)], T0)
    assert match1 is not None
    assert match1.event_type == "restricted_zone_intrusion"
    assert match1.severity == Severity.HIGH
    assert match1.zone_id == "restricted"
    assert match1.zone_name == "Restricted Area"
    assert match1.track_ids == [1]
    assert match1.metadata["dwell_seconds"] == 0.0
    assert match1.metadata["entered_zone_at"] == T0.isoformat()
    assert match1.metadata["track_state"] == "confirmed"

    match2 = evaluate(
        rule, spatial, [VERTICAL_ZONE], [make_track(1, inside, T0 + STEP)], T0 + STEP
    )
    assert match2 is not None
    assert match2.metadata["dwell_seconds"] == 0.5
    assert match2.metadata["entered_zone_at"] == T0.isoformat()


def test_rule_explains_conditions():
    rule = make_rule()
    spatial = SpatioTemporalState()
    inside = BBox(0.45, 0.4, 0.1, 0.3)
    match = evaluate(rule, spatial, [VERTICAL_ZONE], [make_track(1, inside, T0)], T0)
    assert match is not None
    names = {c.name for c in match.conditions}
    assert {"object_class", "zone", "confidence", "in_zone_dwell_seconds"} <= names
    assert all(c.satisfied for c in match.conditions)
    confidence_cond = next(c for c in match.conditions if c.name == "confidence")
    assert confidence_cond.actual == 0.9
    assert confidence_cond.threshold == 0.5


def test_rule_no_match_outside_zone():
    rule = make_rule()
    spatial = SpatioTemporalState()
    outside = BBox(0.05, 0.4, 0.1, 0.3)
    assert evaluate(rule, spatial, [VERTICAL_ZONE], [make_track(1, outside, T0)], T0) is None


def test_rule_respects_min_confidence():
    rule = make_rule()
    spatial = SpatioTemporalState()
    inside = BBox(0.45, 0.4, 0.1, 0.3)
    match = evaluate(
        rule, spatial, [VERTICAL_ZONE], [make_track(1, inside, T0, confidence=0.4)], T0
    )
    assert match is None


def test_rule_respects_class_filter():
    rule = make_rule()
    spatial = SpatioTemporalState()
    inside = BBox(0.45, 0.4, 0.1, 0.3)
    assert (
        evaluate(
            rule,
            spatial,
            [VERTICAL_ZONE],
            [make_track(1, inside, T0, class_name="vehicle")],
            T0,
        )
        is None
    )


def test_rule_respects_zone_filter():
    rule = make_rule(zone_ids=["hazardous"])
    spatial = SpatioTemporalState()
    inside = BBox(0.45, 0.4, 0.1, 0.3)
    assert evaluate(rule, spatial, [VERTICAL_ZONE], [make_track(1, inside, T0)], T0) is None


def test_rule_respects_schedule_window():
    rule = make_rule(
        schedule=Schedule(
            timezone="UTC",
            windows=[ScheduleWindow(days=list(range(7)), start="20:00", end="06:00")],
        )
    )
    spatial = SpatioTemporalState()
    inside = BBox(0.45, 0.4, 0.1, 0.3)
    midday = T0.replace(hour=12)
    night = T0.replace(hour=22)

    assert evaluate(rule, spatial, [VERTICAL_ZONE], [make_track(1, inside, midday)], midday) is None
    night_match = evaluate(
        rule, spatial, [VERTICAL_ZONE], [make_track(1, inside, night)], night
    )
    assert night_match is not None
    assert any(c.name == "schedule" for c in night_match.conditions)


def test_rule_requires_spatial_state():
    rule = make_rule()
    ctx = EvaluationContext(
        camera_id="cam1",
        camera_timezone="UTC",
        ts=T0,
        zones=[VERTICAL_ZONE],
        tracks=[],
        transitions=[],
        spatial=None,
    )
    assert rule.evaluate(ctx) is None
