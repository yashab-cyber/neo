from fastapi import FastAPI, Request, status, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .routers import system, commands, ai, files, tasks, security, metrics, websocket, auth, documentation
from neo.config import settings
from .middleware import RequestContextMiddleware
from .errors import neo_error_handler, unhandled_error_handler, NEOError
from .responses import error_response
from neo.utils.logging import configure_logging


def api_key_auth(request: Request):
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if settings.api_keys and api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title=settings.app_name, version=settings.version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    # Global dependencies list (API key auth)
    # Use JWT verify for main protected endpoints; leave token issuance open.
    from .routers.auth import verify_jwt  # local import to avoid circular
    deps = [Depends(verify_jwt)]

    app.include_router(auth.router, prefix="/api/v1", tags=["auth"])  # public issuance
    app.include_router(system.router, prefix="/api/v1", tags=["system"], dependencies=deps)
    app.include_router(commands.router, prefix="/api/v1", tags=["commands"], dependencies=deps)
    app.include_router(ai.router, prefix="/api/v1", tags=["ai"], dependencies=deps)
    app.include_router(files.router, prefix="/api/v1", tags=["files"], dependencies=deps)
    app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"], dependencies=deps)
    app.include_router(security.router, prefix="/api/v1", tags=["security"], dependencies=deps)
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"], dependencies=deps)
    app.include_router(websocket.router, prefix="/api/v1", tags=["ws"], dependencies=deps)
    app.include_router(documentation.router, prefix="/api/v1", tags=["documentation"], dependencies=deps)

    # Custom exception handlers
    app.add_exception_handler(NEOError, neo_error_handler)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):  # pragma: no cover
        return JSONResponse(
            status_code=422,
            content=error_response(
                "INVALID_PARAMETER",
                "Validation error",
                exc.errors(),
                getattr(request.state, "request_id", None),
            ),
        )

    @app.exception_handler(Exception)
    async def fallback_handler(request: Request, exc: Exception):  # pragma: no cover
        return await unhandled_error_handler(request, exc)

    # Startup / shutdown tasks (ensure command queue worker persists across requests)
    try:  # pragma: no cover - infrastructure wiring
        from neo.services.command_queue import queue

        @app.on_event("startup")
        async def _start_queue():
            await queue.start()

        @app.on_event("shutdown")
        async def _stop_queue():
            await queue.stop()
    except Exception:  # pragma: no cover
        pass

    return app
