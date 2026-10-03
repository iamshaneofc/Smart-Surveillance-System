import json

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from packages.common.logging import get_logger, request_id_var
from packages.common.textutil import redact_secrets
from packages.schemas.common import ErrorDetail, ErrorResponse

log = get_logger(__name__)


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def error_response(
    code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details or {},
            request_id=request_id_var.get(),
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return error_response(exc.code, exc.message, exc.status_code, exc.details)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = json.loads(json.dumps(exc.errors(), default=str))
    for error in errors:
        error.pop("input", None)
    return error_response(
        "validation_error",
        "request validation failed",
        422,
        {"errors": errors},
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception(
        "unhandled error on %s %s: %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
        redact_secrets(exc),
        exc_info=exc,
    )
    return error_response("internal_error", "an unexpected error occurred", 500)
