from datetime import datetime, timedelta, timezone

from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus
from services.alerts.notifiers import LoggingNotifier, MqttNotifier
from services.alerts.router import AlertRouter

TZ = timezone.utc


def _event(ts: datetime, event_type="restricted_zone_intrusion", event_id="evt1") -> Event:
    return Event(
        event_id=event_id,
        camera_id="cam1",
        timestamp=ts,
        event_type=event_type,
        severity=Severity.HIGH,
        status=EventStatus.NEW,
        confidence=0.9,
        created_at=ts,
        updated_at=ts,
    )


def test_dispatch_sends_and_records():
    notifier = LoggingNotifier()
    router = AlertRouter(notifiers={"dashboard": notifier}, default_channels=["dashboard"])
    ts = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    results = router.dispatch(_event(ts))
    assert results[0].status == "sent"
    assert notifier.sent[0].title.startswith("HIGH")
    assert "cam1" in notifier.sent[0].title


def test_cooldown_prevents_storm():
    notifier = LoggingNotifier()
    router = AlertRouter(
        notifiers={"dashboard": notifier},
        default_channels=["dashboard"],
        cooldown_seconds=120.0,
    )
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    assert router.dispatch(_event(t0))[0].status == "sent"
    second = router.dispatch(_event(t0 + timedelta(seconds=10), event_id="evt2"))
    assert second[0].status == "skipped"
    assert second[0].reason == "cooldown"
    third = router.dispatch(_event(t0 + timedelta(seconds=130), event_id="evt3"))
    assert third[0].status == "sent"
    assert len(notifier.sent) == 2


def test_rate_cap_per_camera():
    notifier = LoggingNotifier()
    router = AlertRouter(
        notifiers={"dashboard": notifier},
        default_channels=["dashboard"],
        cooldown_seconds=0.0,
        rate_cap_per_minute=3,
    )
    t0 = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    for i in range(3):
        assert router.dispatch(_event(t0 + timedelta(seconds=i), event_id=f"e{i}"))[0].status == "sent"
    capped = router.dispatch(_event(t0 + timedelta(seconds=10), event_id="e3"))
    assert capped[0].status == "skipped"
    assert capped[0].reason == "rate_cap"


def test_missing_notifier_skipped():
    router = AlertRouter(notifiers={}, default_channels=["email"])
    ts = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    result = router.dispatch(_event(ts))
    assert result[0].status == "skipped"
    assert result[0].reason == "no notifier"


def test_escalation_for_unacknowledged():
    notifier = LoggingNotifier()
    router = AlertRouter(
        notifiers={"dashboard": notifier},
        default_channels=["dashboard"],
        escalation_channel="dashboard",
        escalation_after_seconds=300.0,
    )
    ts = datetime(2026, 6, 15, 22, 0, 0, tzinfo=TZ)
    event = _event(ts)
    now = ts + timedelta(seconds=600)
    results = router.escalation_candidates([event], now)
    assert len(results) == 1
    assert results[0].status == "sent"
    assert notifier.sent[-1].title.startswith("[ESCALATED]")

    event.acknowledged_at = ts + timedelta(seconds=60)
    assert router.escalation_candidates([event], now) == []


def test_mqtt_notifier_is_interface_only():
    try:
        MqttNotifier().send(object())
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
