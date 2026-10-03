from datetime import datetime, timedelta, timezone

from packages.schemas.common import Severity
from packages.schemas.rule import RuleDefinition, RuleType, Schedule, ScheduleWindow
from services.rules.base import RuleSet
from services.rules.context import EvaluationContext, SpatioTemporalState, ZoneContext
from services.tracking.types import TrackState
from services.inference.types import BBox

TZ = timezone.utc
SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]


def _zone(zone_id: str = "restricted") -> ZoneContext:
    return ZoneContext(id=zone_id, name="Restricted Area", zone_type="restricted", polygon=SQUARE)


def _track(track_id: int, cx: float, cy: float, ts: datetime, confidence: float = 0.9) -> TrackState:
    return TrackState(
        track_id=track_id,
        class_name="person",
        confidence=confidence,
        bbox=BBox(cx - 0.05, cy - 0.05, 0.1, 0.1),
        first_seen=ts,
        last_seen=ts,
        state="confirmed",
    )


def _enter_rule(**overrides) -> RuleDefinition:
    payload = dict(
        rule_id="zone-entry",
        name="Zone entry",
        rule_type=RuleType.ZONE_ENTER,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        zone_ids=["restricted"],
        params={"classes": ["person"]},
        confirm_seconds=0.0,
        cooldown_seconds=0.0,
        min_confidence=0.5,
    )
    payload.update(overrides)
    return RuleDefinition(**payload)


def _dwell_rule(**overrides) -> RuleDefinition:
    payload = dict(
        rule_id="zone-dwell",
        name="Zone dwell",
        rule_type=RuleType.ZONE_DWELL,
        event_type="dwell_violation",
        severity=Severity.MEDIUM,
        zone_ids=["restricted"],
        params={"classes": ["person"], "dwell_seconds": 20},
        confirm_seconds=0.0,
        cooldown_seconds=0.0,
        min_confidence=0.5,
    )
    payload.update(overrides)
    return RuleDefinition(**payload)


def _line_rule(**overrides) -> RuleDefinition:
    payload = dict(
        rule_id="line-cross",
        name="Line crossing",
        rule_type=RuleType.LINE_CROSS,
        event_type="line_crossing",
        severity=Severity.MEDIUM,
        line=[(0.5, 0.0), (0.5, 1.0)],
        params={"classes": ["person"]},
        confirm_seconds=0.0,
        cooldown_seconds=0.0,
        min_confidence=0.5,
    )
    payload.update(overrides)
    return RuleDefinition(**payload)


def _ctx(state: SpatioTemporalState, tracks, zones, ts, camera="cam1") -> EvaluationContext:
    enriched = state.enrich(tracks, zones)
    transitions = state.update(enriched, zones, ts)
    return EvaluationContext(
        camera_id=camera,
        camera_timezone="UTC",
        ts=ts,
        zones=zones,
        tracks=enriched,
        transitions=transitions,
        spatial=state,
    )


def test_zone_enter_rule_fires_on_transition_only():
    state = SpatioTemporalState()
    zones = [_zone()]
    ruleset = RuleSet([_enter_rule()])
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)

    outside = _track(1, 0.1, 0.1, t0)
    assert ruleset.evaluate_all(_ctx(state, [outside], zones, t0)) == []

    inside = _track(1, 0.5, 0.5, t0 + timedelta(seconds=1))
    matches = ruleset.evaluate_all(_ctx(state, [inside], zones, t0 + timedelta(seconds=1)))
    assert len(matches) == 1
    match = matches[0]
    assert match.event_type == "restricted_zone_intrusion"
    assert match.track_ids == [1]
    assert match.zone_name == "Restricted Area"
    names = {c.name for c in match.conditions}
    assert {"object_class", "zone", "confidence"} <= names

    still_inside = _track(1, 0.51, 0.5, t0 + timedelta(seconds=2))
    assert ruleset.evaluate_all(_ctx(state, [still_inside], zones, t0 + timedelta(seconds=2))) == []


def test_zone_enter_respects_min_confidence_and_class():
    state = SpatioTemporalState()
    zones = [_zone()]
    ruleset = RuleSet([_enter_rule()])
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)

    low_conf = _track(1, 0.5, 0.5, t0, confidence=0.2)
    assert ruleset.evaluate_all(_ctx(state, [low_conf], zones, t0)) == []

    state2 = SpatioTemporalState()
    vehicle = _track(2, 0.5, 0.5, t0)
    vehicle.class_name = "vehicle"
    assert ruleset.evaluate_all(_ctx(state2, [vehicle], zones, t0)) == []


def test_dwell_rule_requires_duration():
    state = SpatioTemporalState()
    zones = [_zone()]
    ruleset = RuleSet([_dwell_rule()])
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)

    inside = _track(1, 0.5, 0.5, t0)
    ruleset.evaluate_all(_ctx(state, [inside], zones, t0))
    t1 = t0 + timedelta(seconds=10)
    assert ruleset.evaluate_all(_ctx(state, [_track(1, 0.5, 0.5, t1)], zones, t1)) == []

    t2 = t0 + timedelta(seconds=21)
    matches = ruleset.evaluate_all(_ctx(state, [_track(1, 0.5, 0.5, t2)], zones, t2))
    assert len(matches) == 1
    dwell_cond = next(c for c in matches[0].conditions if c.name == "dwell_seconds")
    assert dwell_cond.satisfied is True
    assert dwell_cond.actual >= 21


def test_dwell_resets_after_leaving_zone():
    state = SpatioTemporalState()
    zones = [_zone()]
    ruleset = RuleSet([_dwell_rule()])
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)

    ruleset.evaluate_all(_ctx(state, [_track(1, 0.5, 0.5, t0)], zones, t0))
    outside = t0 + timedelta(seconds=30)
    ruleset.evaluate_all(_ctx(state, [_track(1, 0.1, 0.1, outside)], zones, outside))
    reenter = outside + timedelta(seconds=1)
    assert ruleset.evaluate_all(_ctx(state, [_track(1, 0.5, 0.5, reenter)], zones, reenter)) == []


def test_line_cross_rule_detects_both_directions():
    state = SpatioTemporalState()
    ruleset = RuleSet([_line_rule()])
    zones: list[ZoneContext] = []
    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=TZ)

    ruleset.evaluate_all(_ctx(state, [_track(1, 0.4, 0.5, t0)], zones, t0))
    t1 = t0 + timedelta(seconds=1)
    matches = ruleset.evaluate_all(_ctx(state, [_track(1, 0.6, 0.5, t1)], zones, t1))
    assert len(matches) == 1
    assert matches[0].metadata["direction"] == "left_to_right"


def test_schedule_blocks_rule_outside_window():
    state = SpatioTemporalState()
    zones = [_zone()]
    schedule = Schedule(
        windows=[ScheduleWindow(days=list(range(7)), start="20:00", end="06:00")],
        timezone="UTC",
    )
    ruleset = RuleSet([_enter_rule(schedule=schedule)])
    noon = datetime(2026, 6, 15, 12, 0, 0, tzinfo=TZ)
    assert ruleset.evaluate_all(_ctx(state, [_track(1, 0.5, 0.5, noon)], zones, noon)) == []

    state2 = SpatioTemporalState()
    night = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    matches = ruleset.evaluate_all(_ctx(state2, [_track(1, 0.5, 0.5, night)], zones, night))
    assert len(matches) == 1
    assert any(c.name == "schedule" for c in matches[0].conditions)


def test_ruleset_skips_unimplemented_rule_types():
    unsupported = _enter_rule(
        rule_id="count-rule",
        rule_type=RuleType.OBJECT_COUNT,
        event_type="crowding",
    )
    ruleset = RuleSet([unsupported])
    assert ruleset.rules == []


def test_rule_packs_load_and_build():
    from services.rules.pack import available_packs, load_pack

    packs = available_packs()
    assert len(packs) >= 3
    for path in packs:
        pack = load_pack(path)
        assert pack.rules
        ruleset = RuleSet(pack.rules)
        assert len(ruleset.rules) == len(
            [r for r in pack.rules if r.rule_type in (RuleType.ZONE_ENTER, RuleType.ZONE_DWELL, RuleType.LINE_CROSS, RuleType.RESTRICTED_ZONE_INTRUSION)]
        )

