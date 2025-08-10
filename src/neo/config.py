"""Application configuration using Pydantic BaseSettings.

Loads environment variables for runtime configuration. Provides a singleton
`settings` object that can be imported across modules.
"""
from __future__ import annotations
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    model_config = ConfigDict(env_prefix="NEO_", case_sensitive=False)

    app_name: str = "NEO API"
    version: str = "1.0.0"
    environment: str = Field("dev", description="Environment name: dev|staging|prod")
    api_keys: List[str] = Field(default_factory=lambda: ["devkey"], description="Allowed API keys")
    request_timeout_seconds: int = 30
    log_level: str = "INFO"
    enable_cors: bool = True
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    prometheus_enabled: bool = True
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 20
    rate_limit_window_seconds: int = 60
    rate_limit_backend: str = "memory"  # memory|redis
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = Field("change-me-dev-secret", description="JWT signing secret")
    jwt_algorithm: str = "HS256"
    jwt_exp_minutes: int = 60
    jwt_refresh_exp_minutes: int = 60 * 24
    security_headers_enabled: bool = True
    default_role: str = "user"
    role_scopes: dict[str, list[str]] = {
        "user": ["read:status", "read:chat"],
        "admin": [
            "read:status",
            "read:chat",
            "write:commands",
            "manage:tasks",
            "manage:security",
            "manage:cognitive",
            "read:knowledge",
            "write:knowledge",
            "read:memory",
            "write:memory",
        ],
    }
    database_url: str = Field("sqlite+aiosqlite:///./data/app.db", description="SQLAlchemy database URL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def override_settings(**overrides):  # test helper
    if get_settings.cache_info().currsize:  # type: ignore[attr-defined]
        get_settings.cache_clear()  # type: ignore[attr-defined]
    # Create new settings with environment + overrides
    return Settings(**overrides)
