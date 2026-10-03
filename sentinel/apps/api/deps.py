from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from apps.api.errors import AppError
from apps.api.security import UnauthorizedError, resolve_principal
from packages.config import Settings
from packages.db.base import get_session
from packages.schemas.auth import Principal


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_bus(request: Request):
    return request.app.state.bus


def get_db_session() -> Generator[Session, None, None]:
    yield from get_session()


def get_current_principal(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Principal:
    settings: Settings = request.app.state.settings
    try:
        principal = resolve_principal(settings.auth_mode, settings.auth_api_keys, x_api_key)
    except UnauthorizedError as exc:
        raise AppError("unauthorized", str(exc), 401) from exc
    if settings.auth_mode == "disabled":
        request.app.state._warned_auth_disabled = getattr(
            request.app.state, "_warned_auth_disabled", False
        )
    return principal


def require_permission(permission: str):
    def dependency(principal: Annotated[Principal, Depends(get_current_principal)]) -> Principal:
        if not principal.has(permission):
            raise AppError(
                "forbidden",
                f"missing permission: {permission}",
                403,
            )
        return principal

    return dependency


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
PrincipalDep = Annotated[Principal, Depends(get_current_principal)]
DbDep = Annotated[Session, Depends(get_db_session)]
