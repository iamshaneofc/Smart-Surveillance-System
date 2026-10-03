from datetime import datetime, timedelta, timezone

from packages.schemas.rule import Schedule, ScheduleWindow
from services.rules.geometry import crossing_direction, point_in_polygon, segments_intersect

SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]


def test_point_in_polygon_inside_and_outside():
    assert point_in_polygon((0.5, 0.5), SQUARE) is True
    assert point_in_polygon((0.1, 0.1), SQUARE) is False
    assert point_in_polygon((0.9, 0.5), SQUARE) is False


def test_point_in_polygon_degenerate():
    assert point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 1.0)]) is False


def test_segments_intersect():
    assert segments_intersect((0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0)) is True
    assert segments_intersect((0.0, 0.0), (0.1, 0.1), (0.5, 0.5), (0.6, 0.6)) is False


def test_crossing_direction_both_ways():
    line_a, line_b = (0.5, 0.0), (0.5, 1.0)
    assert crossing_direction((0.4, 0.5), (0.6, 0.5), line_a, line_b) == "left_to_right"
    assert crossing_direction((0.6, 0.5), (0.4, 0.5), line_a, line_b) == "right_to_left"
    assert crossing_direction((0.4, 0.5), (0.45, 0.5), line_a, line_b) is None


def test_schedule_empty_always_active():
    at = datetime(2026, 6, 15, 13, 0, tzinfo=timezone.utc)
    assert Schedule() == Schedule(windows=[])
    schedule = Schedule(windows=[])
    from services.rules.schedule import is_active

    assert is_active(schedule, at, "UTC") is True


def test_schedule_window_active_and_inactive():
    from services.rules.schedule import is_active

    night = Schedule(
        windows=[ScheduleWindow(days=list(range(7)), start="20:00", end="06:00")],
        timezone="UTC",
    )
    late = datetime(2026, 6, 15, 22, 0, tzinfo=timezone.utc)
    noon = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    early = datetime(2026, 6, 15, 5, 0, tzinfo=timezone.utc)
    assert is_active(night, late, "UTC") is True
    assert is_active(night, early, "UTC") is True
    assert is_active(night, noon, "UTC") is False


def test_schedule_day_filter():
    from services.rules.schedule import is_active

    weekdays = Schedule(
        windows=[ScheduleWindow(days=[0, 1, 2, 3, 4], start="08:00", end="18:00")],
        timezone="UTC",
    )
    monday_noon = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    assert monday_noon.weekday() == 0
    assert is_active(weekdays, monday_noon, "UTC") is True
    saturday_noon = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)
    assert saturday_noon.weekday() == 5
    assert is_active(weekdays, saturday_noon, "UTC") is False
