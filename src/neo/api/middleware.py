"""Custom middleware for request context & logging."""
from __future__ import annotations
import time, uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from neo.utils.logging import get_logger
from neo.config import settings
from fastapi import HTTPException, status
from collections import defaultdict, deque
from time import time as now
from prometheus_client import Counter, Histogram
import asyncio
try:  # optional Redis backend
    from redis import asyncio as redis_async  # type: ignore
except Exception:  # pragma: no cover - redis not installed scenario
    redis_async = None  # type: ignore

REQUEST_COUNT = Counter("neo_http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("neo_http_request_duration_seconds", "Request latency", ["method", "path"])
RATE_LIMIT_REJECTIONS = Counter("neo_rate_limit_rejections_total", "Total rate limited requests", ["client_ip"])

# NOTE:
# Buckets are now stored on the FastAPI app state (app.state.rate_limit_buckets)
# to avoid cross-test interference when multiple TestClient instances are created.
# The old module-level _buckets is preserved only as a fallback if middleware is
# used outside the application factory (should not happen in normal flow).
_fallback_buckets: dict[str, deque[float]] = defaultdict(lambda: deque())

log = get_logger("middleware")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        # Rate limiting (IP-based) with optional Redis backend
        client_ip = request.client.host if request.client else "unknown"
        # Resolve bucket store (per-app if available)
        bucket_store: dict[str, deque[float]] = getattr(
            request.app.state, "rate_limit_buckets", _fallback_buckets
        )
        ratelimit_response: Response | None = None
        if settings.rate_limit_backend == "redis" and redis_async:
            try:
                if not hasattr(request.app.state, "redis_rl"):
                    request.app.state.redis_rl = redis_async.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)  # type: ignore[attr-defined]
                key = f"rl:{client_ip}:{int(now() // settings.rate_limit_window_seconds)}"
                pipe = request.app.state.redis_rl.pipeline()  # type: ignore[attr-defined]
                pipe.incr(key)
                pipe.expire(key, settings.rate_limit_window_seconds)
                current_count, _ = await pipe.execute()
                if int(current_count) > settings.rate_limit_per_minute:
                    RATE_LIMIT_REJECTIONS.labels(client_ip).inc()
                    raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
            except Exception:  # pragma: no cover - fallback to memory
                pass
        if ratelimit_response is None and (settings.rate_limit_backend == "memory" or not redis_async):
            window = settings.rate_limit_window_seconds
            limit = settings.rate_limit_per_minute
            bucket = bucket_store[client_ip]
            current = now()
            while bucket and current - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= limit:
                RATE_LIMIT_REJECTIONS.labels(client_ip).inc()
                # Return a response instead of raising to avoid BaseHTTPMiddleware task group issues
                ratelimit_response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded"},
                )
            else:
                bucket.append(current)
        response: Response | None = None
        try:
            if ratelimit_response is not None:
                response = ratelimit_response
            else:
                with REQUEST_LATENCY.labels(request.method, request.url.path).time():
                    response = await call_next(request)
            return response
        finally:
            status_code = getattr(response, "status_code", 0)
            REQUEST_COUNT.labels(request.method, request.url.path, status_code).inc()
            duration_ms = (time.perf_counter() - start) * 1000
            if settings.security_headers_enabled and response is not None:
                response.headers.setdefault("X-Content-Type-Options", "nosniff")
                response.headers.setdefault("X-Frame-Options", "DENY")
                response.headers.setdefault("X-XSS-Protection", "1; mode=block")
                response.headers.setdefault("Referrer-Policy", "no-referrer")
                response.headers.setdefault("Content-Security-Policy", "default-src 'self';")
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(duration_ms, 2),
                request_id=request_id,
            )
