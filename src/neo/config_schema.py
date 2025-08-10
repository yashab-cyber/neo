from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional

class DeepLearningConfig(BaseModel):
    enabled: bool = True
    model_path: str
    inference_timeout: int = Field(ge=1)

class NeuroLearningConfig(BaseModel):
    enabled: bool = True
    plasticity_rate: float = Field(ge=0, le=1)
    adaptation_threshold: float = Field(ge=0, le=1)

class RecursiveLearningConfig(BaseModel):
    enabled: bool = True
    recursion_depth: int = Field(ge=1, le=1000)
    convergence_threshold: float = Field(ge=0, le=1)

class LearningParadigms(BaseModel):
    deep_learning: DeepLearningConfig
    neuro_learning: NeuroLearningConfig
    recursive_learning: RecursiveLearningConfig

class MemorySystemConfig(BaseModel):
    short_term_capacity: int = Field(gt=0, le=100)
    working_memory_chunks: int = Field(gt=0, le=32)
    consolidation_interval: str
    forgetting_curve: str

class AIEngineConfig(BaseModel):
    learning_paradigms: LearningParadigms
    memory_system: MemorySystemConfig

class AuthenticationConfig(BaseModel):
    method: str
    session_timeout: str
    max_failed_attempts: int
    lockout_duration: str

class ThreatDetectionConfig(BaseModel):
    enabled: bool
    sensitivity: str
    response_mode: str
    alert_threshold: float

class SecurityConfig(BaseModel):
    authentication: AuthenticationConfig
    threat_detection: ThreatDetectionConfig

class CoreConfig(BaseModel):
    version: str
    environment: str
    debug_mode: bool
    log_level: str
    max_concurrent_sessions: int

class SystemConfig(BaseModel):
    core: CoreConfig
    ai_engine: AIEngineConfig
    security: SecurityConfig

class InterfacePreferences(BaseModel):
    theme: str
    language: str
    voice_enabled: bool
    notifications: bool

class AIBehaviorPreferences(BaseModel):
    interaction_style: str
    verbosity_level: str
    learning_rate: str
    personalization_enabled: bool

class PrivacyPreferences(BaseModel):
    data_collection: str
    analytics_enabled: bool
    telemetry_enabled: bool
    sharing_permissions: list

class UserPreferences(BaseModel):
    interface: InterfacePreferences
    ai_behavior: AIBehaviorPreferences
    privacy: PrivacyPreferences

class RootConfig(BaseModel):
    system_config: SystemConfig
    user_preferences: UserPreferences

    @field_validator("system_config")
    @classmethod
    def validate_environment(cls, v: SystemConfig):
        if v.core.environment not in {"production", "staging", "dev", "test"}:
            raise ValueError("invalid environment")
        return v


def validate_config(data: dict[str, Any]) -> RootConfig:
    return RootConfig(**data)

__all__ = ["validate_config", "RootConfig"]
