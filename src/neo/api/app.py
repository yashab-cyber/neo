from fastapi import FastAPI, Request, status, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .routers import system, commands, ai, files, tasks, security, metrics, websocket, auth, documentation, knowledge, memory
from .routers import cognitive
from neo.config import settings
from .middleware import RequestContextMiddleware
from .errors import neo_error_handler, unhandled_error_handler, NEOError
from .responses import error_response
from neo.utils.logging import configure_logging
import time
from contextlib import asynccontextmanager
from neo.db import Base, engine
from collections import defaultdict, deque

_app_start_time = time.time()


def api_key_auth(request: Request):
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if settings.api_keys and api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    # startup
    try:
        from neo.services.command_queue import queue
        await queue.start()
    except Exception:
        pass
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception:
        pass
    yield
    # shutdown
    try:
        from neo.services.command_queue import queue
        await queue.stop()
    except Exception:
        pass


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    # Per-app rate limit buckets (IP -> deque[timestamps]) to isolate tests / instances
    app.state.rate_limit_buckets = defaultdict(lambda: deque())  # type: ignore[attr-defined]

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
    app.include_router(cognitive.router, prefix="/api/v1", tags=["cognitive"], dependencies=deps)
    app.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"], dependencies=deps)
    app.include_router(memory.router, prefix="/api/v1", tags=["memory"], dependencies=deps)

    # Public root & health endpoints ---------------------------------
    @app.get("/")
    async def root():  # pragma: no cover - simple informational endpoint
        uptime_s = int(time.time() - _app_start_time)
        return {
            "app": settings.app_name,
            "version": settings.version,
            "message": "NEO API root. See /docs for interactive API, /api/v1/status for system status (auth required).",
            "uptime_seconds": uptime_s,
        }

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():  # pragma: no cover
        return {"ready": True, "version": settings.version}

    @app.get("/favicon.ico")
    async def favicon():  # pragma: no cover
        return JSONResponse(status_code=204, content=None)

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

    return app
