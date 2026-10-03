import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from apps.api.deps import get_app_settings, get_bus, get_db_session
from packages.config import Settings

router = APIRouter(tags=["system"])


@router.get("/system/health")
def system_health(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    bus=Depends(get_bus),
    session: Session = Depends(get_db_session),
):
    checks: dict[str, dict] = {}

    try:
        session.execute(text("SELECT 1"))
        checks["database"] = {"status": "healthy", "dialect": session.bind.dialect.name}
    except Exception as exc:
        checks["database"] = {"status": "error", "detail": str(exc)[:200]}

    try:
        bus_ok = bus.healthcheck()
        checks["event_bus"] = {"status": "healthy" if bus_ok else "error", "backend": settings.bus_url.split("://")[0]}
    except Exception as exc:
        checks["event_bus"] = {"status": "error", "detail": str(exc)[:200]}

    statuses = [c["status"] for c in checks.values()]
    if all(s == "healthy" for s in statuses):
        overall, status_code = "healthy", 200
    elif any(s == "healthy" for s in statuses):
        overall, status_code = "degraded", 503
    else:
        overall, status_code = "error", 503

    started_at = getattr(request.app.state, "started_at", time.time())
    body = {
        "status": overall,
        "checks": checks,
        "app": {
            "name": settings.app_name,
            "version": settings.version,
            "env": settings.env,
            "uptime_seconds": round(time.time() - started_at, 1),
        },
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }
    return JSONResponse(status_code=status_code, content=body)
