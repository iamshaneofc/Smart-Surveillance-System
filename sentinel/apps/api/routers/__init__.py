from fastapi import APIRouter

from apps.api.routers import (
    alerts,
    auth,
    cameras,
    events,
    evidence,
    health,
    rules,
    stream,
    zones,
)
from apps.api.routers.stubs import models_router, search_router

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(cameras.router)
api_router.include_router(events.router)
api_router.include_router(evidence.router)
api_router.include_router(zones.zones_router)
api_router.include_router(zones.camera_zones_router)
api_router.include_router(rules.router)
api_router.include_router(alerts.router)
api_router.include_router(stream.router)
api_router.include_router(models_router)
api_router.include_router(search_router)
