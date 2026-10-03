import threading
from collections import deque
from datetime import datetime, timedelta

from packages.common.timeutil import utcnow
from packages.schemas.auth import Principal, Role, permissions_for


class UnauthorizedError(Exception):
    pass


class RateLimitedError(Exception):
    pass


def resolve_principal(auth_mode: str, api_keys, provided_key: str | None) -> Principal:
    if auth_mode == "disabled":
        roles = [Role.ADMIN]
        return Principal(
            user="dev-admin",
            roles=roles,
            permissions=sorted(permissions_for(roles)),
            auth_mode="disabled",
        )
    if not provided_key:
        raise UnauthorizedError("missing X-API-Key header")
    for spec in api_keys:
        if spec.key == provided_key:
            roles = [Role(r) for r in spec.roles] or [Role.VIEWER]
            return Principal(
                user=spec.user,
                roles=roles,
                permissions=sorted(permissions_for(roles)),
                auth_mode="api_key",
            )
    raise UnauthorizedError("invalid API key")


class SlidingWindowLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[datetime]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: datetime | None = None) -> bool:
        if self.limit <= 0:
            return True
        now = now or utcnow()
        window_start = now - timedelta(minutes=1)
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True
