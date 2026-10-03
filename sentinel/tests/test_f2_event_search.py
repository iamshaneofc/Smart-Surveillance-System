from datetime import datetime, timedelta, timezone

from packages.schemas.common import Severity
from packages.schemas.event import Event, EventStatus
from services.events.repository import SqlEventRepository

NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def _seed(session, event_id, timestamp, rule_id="restricted-zone-entry", camera_id="cam1"):
    event = Event(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=timestamp,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        status=EventStatus.NEW,
        confidence=0.9,
        zone_id="z1",
        zone_name="Restricted",
        rule_id=rule_id,
        rule_name="Restricted zone entry",
        model_versions={"detector": "stub:1"},
        created_at=timestamp,
        updated_at=timestamp,
    )
    SqlEventRepository(session).save(event)
    return event


def _seed_range():
    from packages.db import base as db_base

    with db_base.session_scope() as scoped:
        for i in range(7):
            _seed(
                scoped,
                f"evt_p{i}",
                NOW + timedelta(minutes=i),
                rule_id="rule-a" if i % 2 == 0 else "rule-b",
            )
        _seed(scoped, "evt_tie1", NOW - timedelta(hours=1), rule_id="rule-a")
        _seed(scoped, "evt_tie2", NOW - timedelta(hours=1), rule_id="rule-a")


def test_event_filter_by_rule_id(client):
    _seed_range()
    response = client.get("/api/v1/events", params={"rule_id": "rule-b"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert all(item["rule_id"] == "rule-b" for item in body["items"])

    empty = client.get("/api/v1/events", params={"rule_id": "no-such-rule"})
    assert empty.json()["total"] == 0


def test_event_time_range_aliases(client):
    _seed_range()

    start = (NOW + timedelta(minutes=2)).isoformat()
    end = (NOW + timedelta(minutes=4)).isoformat()

    aliases = client.get("/api/v1/events", params={"start_time": start, "end_time": end})
    assert aliases.status_code == 200
    assert aliases.json()["total"] == 3

    canonical = client.get("/api/v1/events", params={"since": start, "until": end})
    assert canonical.json()["total"] == 3

    single = client.get("/api/v1/events", params={"start_time": start})
    assert single.json()["total"] == 5


def test_event_pagination_is_deterministic(client):
    _seed_range()

    first_page = client.get("/api/v1/events", params={"limit": 3, "offset": 0})
    second_page = client.get("/api/v1/events", params={"limit": 3, "offset": 0})
    ids_page1 = [e["event_id"] for e in first_page.json()["items"]]
    ids_page2 = [e["event_id"] for e in second_page.json()["items"]]
    assert ids_page1 == ids_page2
    assert first_page.json()["total"] == 9

    seen = list(ids_page1)
    for offset in (3, 6):
        page = client.get("/api/v1/events", params={"limit": 3, "offset": offset})
        page_ids = [e["event_id"] for e in page.json()["items"]]
        assert not set(page_ids) & set(seen), f"overlap at offset {offset}"
        seen.extend(page_ids)
    assert len(seen) == 9

    timestamps = [
        e["timestamp"]
        for e in first_page.json()["items"] + client.get(
            "/api/v1/events", params={"limit": 6, "offset": 3}
        ).json()["items"]
    ]
    assert timestamps == sorted(timestamps, reverse=True)

    tie_naive = (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()
    tie = [
        e["event_id"]
        for e in first_page.json()["items"] + client.get(
            "/api/v1/events", params={"limit": 6, "offset": 3}
        ).json()["items"]
        if e["timestamp"] == tie_naive
    ]
    assert tie == ["evt_tie2", "evt_tie1"]


def test_event_combined_filters(client):
    _seed_range()
    response = client.get(
        "/api/v1/events",
        params={
            "rule_id": "rule-a",
            "severity": "high",
            "status": "new",
            "camera_id": "cam1",
            "start_time": NOW.isoformat(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert all(item["rule_id"] == "rule-a" for item in body["items"])
