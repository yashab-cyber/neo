from __future__ import annotations
from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import uuid
import jwt
from neo.config import settings
from typing import Set
try:
    from redis import asyncio as redis_async  # type: ignore
except Exception:  # pragma: no cover
    redis_async = None  # type: ignore

_revoked_jti: Set[str] = set()

router = APIRouter()


class AuthRequest(BaseModel):
    api_key: str
    role: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int


async def _is_revoked(jti: str) -> bool:
    if jti in _revoked_jti:
        return True
    if settings.rate_limit_backend == "redis" and redis_async:
        try:
            if not hasattr(router, "redis_rev"):
                router.redis_rev = redis_async.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)  # type: ignore[attr-defined]
            return await router.redis_rev.sismember("revoked_jti", jti)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            return jti in _revoked_jti
    return False


def verify_jwt(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        request.state.subject = payload.get("sub")
        jti = payload.get("jti")
        request.state.jti = jti
        # Synchronous check fallback; for redis async we accept small race here.
        if jti and jti in _revoked_jti:
            raise HTTPException(status_code=401, detail="Revoked token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


def _create_tokens(subject: str, role: str):
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.jwt_exp_minutes)
    jti_access = uuid.uuid4().hex
    scopes = settings.role_scopes.get(role, [])
    access = jwt.encode({"sub": subject, "role": role, "scopes": scopes, "iat": int(now.timestamp()), "exp": int(exp.timestamp()), "type": "access", "jti": jti_access}, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    refresh_exp = now + timedelta(minutes=settings.jwt_refresh_exp_minutes)
    jti_refresh = uuid.uuid4().hex
    refresh = jwt.encode({"sub": subject, "role": role, "iat": int(now.timestamp()), "exp": int(refresh_exp.timestamp()), "type": "refresh", "jti": jti_refresh}, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return access, refresh


@router.post("/auth/token", response_model=TokenResponse)
async def issue_token(req: AuthRequest):
    if req.api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    role = req.role or settings.default_role
    access, refresh = _create_tokens(req.api_key, role)
    return TokenResponse(access_token=access, refresh_token=refresh, expires_in=settings.jwt_exp_minutes * 60)


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest):
    try:
        payload = jwt.decode(req.refresh_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Not a refresh token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    sub = payload.get("sub")
    role = payload.get("role", settings.default_role)
    access, refresh_new = _create_tokens(sub, role)
    return TokenResponse(access_token=access, refresh_token=refresh_new, expires_in=settings.jwt_exp_minutes * 60)


def require_scope(required: str):
    def checker(request: Request):
        token_scopes = getattr(request.state, "token_scopes", None)
        if token_scopes is None:
            # decode again (for simplicity)
            auth = request.headers.get("Authorization", "")
            token = auth.split(" ", 1)[1]
            try:
                payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
                token_scopes = payload.get("scopes", [])
                jti = payload.get("jti")
                if jti and jti in _revoked_jti:
                    raise HTTPException(status_code=401, detail="Revoked token")
            except jwt.PyJWTError:
                raise HTTPException(status_code=401, detail="Invalid token")
            request.state.token_scopes = token_scopes
        if required not in token_scopes:
            raise HTTPException(status_code=403, detail="Insufficient scope")
        return True
    return checker


class RevokeRequest(BaseModel):
    refresh_token: str | None = None


@router.post("/auth/revoke")
async def revoke_token(request: Request, body: RevokeRequest | None = None):
    """Revoke current access token (and optional refresh)"""
    jti = getattr(request.state, "jti", None)
    if not jti:
        # Attempt decode to get jti
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1]
            try:
                payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
                jti = payload.get("jti")
            except jwt.PyJWTError:
                pass
    if not jti:
        raise HTTPException(status_code=400, detail="Missing JTI")
    _revoked_jti.add(jti)
    # Revoke refresh if provided
    if body and body.refresh_token:
        try:
            payload = jwt.decode(body.refresh_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
            rjti = payload.get("jti")
            if rjti:
                _revoked_jti.add(rjti)
        except jwt.PyJWTError:
            pass
    # persist to redis if configured
    if settings.rate_limit_backend == "redis" and redis_async:
        try:
            if not hasattr(router, "redis_rev"):
                router.redis_rev = redis_async.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)  # type: ignore[attr-defined]
            await router.redis_rev.sadd("revoked_jti", jti)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
    return {"revoked": True}
