from datetime import datetime, timedelta, timezone

from packages.schemas.common import Severity
from packages.schemas.event import ConditionEvidence, EventStatus
from packages.schemas.rule import RuleDefinition, RuleType
from services.events.engine import EventEngine, dedup_key_for
from services.rules.base import RuleMatch

TZ = timezone.utc


def _rule(rule_id="r1", confirm=1.0, cooldown=60.0) -> RuleDefinition:
    return RuleDefinition(
        rule_id=rule_id,
        name="Zone entry",
        rule_type=RuleType.ZONE_ENTER,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        confirm_seconds=confirm,
        cooldown_seconds=cooldown,
    )


def _match(ts: datetime, zone_id: str | None = "z1", event_type="restricted_zone_intrusion", confidence=0.9) -> RuleMatch:
    return RuleMatch(
        rule_key="r1",
        rule_name="Zone entry",
        event_type=event_type,
        severity=Severity.HIGH,
        camera_id="cam1",
        ts=ts,
        confidence=confidence,
        track_ids=[1],
        zone_id=zone_id,
        zone_name="Restricted",
        conditions=[
            ConditionEvidence(name="zone", operator="in", actual="Restricted", threshold="z1", satisfied=True)
        ],
    )


def test_temporal_confirmation_required():
    engine = EventEngine(rules={"r1": _rule(confirm=2.0)})
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    assert engine.confirm(_match(t0)) is None
    assert engine.confirm(_match(t0 + timedelta(seconds=1))) is None
    event = engine.confirm(_match(t0 + timedelta(seconds=3)))
    assert event is not None
    assert event.status == EventStatus.NEW
    assert event.severity == Severity.HIGH
    assert event.conditions


def test_zero_confirm_creates_immediately():
    engine = EventEngine(rules={"r1": _rule(confirm=0.0)})
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    assert engine.confirm(_match(t0)) is not None


def test_dedup_while_event_open():
    engine = EventEngine(rules={"r1": _rule(confirm=0.0)})
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    assert engine.confirm(_match(t0)) is not None
    assert engine.confirm(_match(t0 + timedelta(seconds=5))) is None
    assert len(engine.active_events()) == 1


def test_cooldown_after_resolution():
    engine = EventEngine(rules={"r1": _rule(confirm=0.0, cooldown=60.0)})
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    event = engine.confirm(_match(t0))
    engine.transition(event, EventStatus.RESOLVED, at=t0 + timedelta(seconds=5))
    assert engine.confirm(_match(t0 + timedelta(seconds=30))) is None
    assert engine.confirm(_match(t0 + timedelta(seconds=200))) is not None


def test_different_zone_not_deduped():
    engine = EventEngine(rules={"r1": _rule(confirm=0.0)})
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    assert engine.confirm(_match(t0, zone_id="z1")) is not None
    assert engine.confirm(_match(t0, zone_id="z2")) is not None


def test_escalation_on_repeated_events():
    engine = EventEngine(rules={"r1": _rule(confirm=0.0, cooldown=10.0)}, escalation_repeats=3)
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    for i in range(3):
        ts = t0 + timedelta(seconds=i * 120)
        event = engine.confirm(_match(ts))
        assert event is not None
        if i < 2:
            assert event.severity == Severity.HIGH
        else:
            assert event.severity == Severity.CRITICAL
        engine.transition(event, EventStatus.RESOLVED, at=ts + timedelta(seconds=1))


def test_state_machine_transitions():
    engine = EventEngine(rules={"r1": _rule(confirm=0.0)})
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    event = engine.confirm(_match(t0))
    engine.transition(event, EventStatus.ACKNOWLEDGED, actor="op1", at=t0 + timedelta(seconds=1))
    assert event.acknowledged_by == "op1"
    engine.transition(event, EventStatus.RESOLVED, actor="op1", at=t0 + timedelta(seconds=2))
    assert event.resolved_at is not None
    try:
        engine.transition(event, EventStatus.ACKNOWLEDGED, at=t0 + timedelta(seconds=3))
        raise AssertionError("expected invalid transition error")
    except ValueError:
        pass


def test_pending_expires_after_ttl():
    engine = EventEngine(rules={"r1": _rule(confirm=100.0)}, pending_ttl_seconds=10.0)
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    assert engine.confirm(_match(t0)) is None
    assert engine.confirm(_match(t0 + timedelta(seconds=100))) is None
    assert engine.confirm(_match(t0 + timedelta(seconds=150))) is None


def test_dedup_key_stable():
    assert dedup_key_for("c1", "evt", "z1", [3, 1]) == dedup_key_for("c1", "evt", "z1", [1, 3])
    assert dedup_key_for("c1", "evt", "z1", [1]) != dedup_key_for("c1", "evt", "z2", [1])
