"""Custom exceptions & handlers for consistent error responses."""
from __future__ import annotations
from fastapi import Request, status
from fastapi.responses import JSONResponse
from .responses import error_response
from neo.utils.logging import get_logger

log = get_logger("errors")


class NEOError(Exception):
    def __init__(self, code: str, message: str, details: dict | list | None = None, http_status: int = 400):
        self.code = code
        self.message = message
        self.details = details
        self.http_status = http_status
        super().__init__(message)


async def neo_error_handler(request: Request, exc: NEOError):  # pragma: no cover - framework path
    return JSONResponse(
        status_code=exc.http_status,
        content=error_response(exc.code, exc.message, exc.details, getattr(request.state, "request_id", None)),
    )


async def unhandled_error_handler(request: Request, exc: Exception):  # pragma: no cover
    log.exception("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response("INTERNAL_ERROR", "Unexpected server error", None, getattr(request.state, "request_id", None)),
    )
