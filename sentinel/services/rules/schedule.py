from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from packages.schemas.rule import Schedule, ScheduleWindow


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def _window_active(window: ScheduleWindow, local: datetime) -> bool:
    if local.weekday() not in window.days:
        return False
    start = _minutes(window.start)
    end = _minutes(window.end)
    now = local.hour * 60 + local.minute
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def is_active(schedule: Schedule, at: datetime, camera_timezone: str = "UTC") -> bool:
    if not schedule.windows:
        return True
    tz = _resolve_tz(schedule.timezone or camera_timezone)
    local = at.astimezone(tz) if at.tzinfo else at.replace(tzinfo=_UTC).astimezone(tz)
    return any(_window_active(w, local) for w in schedule.windows)


_UTC = timezone.utc


def _resolve_tz(name: str):
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def next_window_label(schedule: Schedule) -> str | None:
    if not schedule.windows:
        return None
    return f"{schedule.windows[0].start}-{schedule.windows[0].end}"
