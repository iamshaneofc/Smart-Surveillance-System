from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.VIEWER: {"events:read", "cameras:read", "evidence:read", "health:read"},
    Role.OPERATOR: {
        "events:read",
        "events:ack",
        "events:dismiss",
        "cameras:read",
        "evidence:read",
        "health:read",
        "zones:read",
        "rules:read",
        "alerts:read",
    },
    Role.ADMIN: {
        "events:read",
        "events:ack",
        "events:dismiss",
        "cameras:read",
        "cameras:manage",
        "evidence:read",
        "evidence:export",
        "health:read",
        "zones:read",
        "zones:manage",
        "rules:read",
        "rules:manage",
        "alerts:read",
        "models:read",
        "models:manage",
        "users:manage",
        "audit:read",
        "system:manage",
    },
}


def permissions_for(roles: list[Role]) -> set[str]:
    perms: set[str] = set()
    for role in roles:
        perms |= ROLE_PERMISSIONS.get(role, set())
    return perms


class Principal(BaseModel):
    user: str
    roles: list[Role] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    org_id: str | None = None
    auth_mode: str = "disabled"

    def has(self, permission: str) -> bool:
        return permission in self.permissions


class WhoAmI(Principal):
    warnings: list[str] = Field(default_factory=list)
