from fastapi import APIRouter, Depends

from apps.api.deps import require_permission
from apps.api.errors import AppError

models_router = APIRouter(prefix="/models", tags=["models"])
search_router = APIRouter(prefix="/search", tags=["search"])


@models_router.get("", status_code=501)
def list_models(principal=Depends(require_permission("models:read"))):
    raise AppError("not_implemented", "model registry API arrives in phase F5", 501)


@search_router.get("", status_code=501)
def search(principal=Depends(require_permission("events:read"))):
    raise AppError("not_implemented", "event search arrives in phase F4", 501)
