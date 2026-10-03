from fastapi import APIRouter, Depends

from apps.api.deps import get_app_settings, get_current_principal
from packages.schemas.auth import Principal, WhoAmI

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=WhoAmI)
def whoami(
    principal: Principal = Depends(get_current_principal),
    settings=Depends(get_app_settings),
):
    warnings: list[str] = []
    if settings.auth_mode == "disabled":
        warnings.append("authentication is disabled (development mode)")
    return WhoAmI(
        user=principal.user,
        roles=principal.roles,
        permissions=principal.permissions,
        org_id=principal.org_id,
        auth_mode=principal.auth_mode,
        warnings=warnings,
    )


@router.post("/login", status_code=501)
def login():
    from apps.api.errors import AppError

    raise AppError(
        "not_implemented",
        "interactive login/OIDC arrives in phase F4; use X-API-Key for now",
        501,
    )
