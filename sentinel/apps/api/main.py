import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from apps.api.errors import (
    app_error_handler,
    error_response,
    unhandled_error_handler,
    validation_error_handler,
    AppError,
)
from apps.api.security import SlidingWindowLimiter
from packages.common.bus import create_bus
from packages.common.ids import new_id
from packages.common.logging import get_logger, request_id_var, setup_logging
from packages.config import Settings, get_settings

log = get_logger(__name__)

HEALTH_PATHS = ("/api/v1/system/health", "/health", "/docs", "/openapi.json")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.auth_mode == "disabled":
            log.warning("authentication is DISABLED - local development only")
        from packages.db import base as db_base

        if not db_base.is_configured():
            db_base.configure(settings.database_url)
        if settings.env != "prod":
            db_base.create_all()
        log.info(
            "sentinel api starting",
            extra={"version": settings.version, "env": settings.env, "bus": settings.bus_url},
        )
        runner = None
        if settings.pipeline.enabled:
            try:
                from services.pipeline.runner import PipelineRunner

                runner = PipelineRunner(settings, bus=app.state.bus)
                runner.start()
            except Exception:
                log.exception("pipeline failed to start - API continues without it")
                runner = None
        app.state.runner = runner
        yield
        if runner is not None:
            try:
                runner.stop()
            except Exception:
                log.exception("pipeline shutdown failed")
        app.state.bus.close()
        log.info("sentinel api stopped")

    app = FastAPI(
        title="SENTINEL",
        description="AI-powered video surveillance & security intelligence platform (foundation phase)",
        version=settings.version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.bus = create_bus(settings.bus_url)
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    app.state.started_at = time.time()

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_id("req")
        token = request_id_var.set(rid)
        try:
            path = request.url.path
            if path.startswith("/api/v1") and path not in HEALTH_PATHS:
                key = request.headers.get("X-API-Key") or (
                    request.client.host if request.client else "unknown"
                )
                if not app.state.limiter.allow(key):
                    return error_response(
                        "rate_limited",
                        "too many requests",
                        429,
                        {"limit_per_minute": settings.rate_limit_per_minute},
                    )
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)

    from apps.api.routers import api_router

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    def root():
        return {
            "name": settings.app_name,
            "version": settings.version,
            "phase": "foundation",
            "docs": "/docs",
            "api": "/api/v1",
        }

    return app


app = create_app()
