from neo.config_schema import validate_config

example = {
  "system_config": {
    "core": {
      "version": "1.0.0",
      "environment": "production",
      "debug_mode": False,
      "log_level": "INFO",
      "max_concurrent_sessions": 1000
    },
    "ai_engine": {
      "learning_paradigms": {
        "deep_learning": {"enabled": True, "model_path": "/models/deep_learning", "inference_timeout": 5000},
        "neuro_learning": {"enabled": True, "plasticity_rate": 0.1, "adaptation_threshold": 0.8},
        "recursive_learning": {"enabled": True, "recursion_depth": 10, "convergence_threshold": 0.95}
      },
      "memory_system": {"short_term_capacity": 7, "working_memory_chunks": 4, "consolidation_interval": "1h", "forgetting_curve": "exponential"}
    },
    "security": {
      "authentication": {"method": "multi_factor", "session_timeout": "24h", "max_failed_attempts": 3, "lockout_duration": "15m"},
      "threat_detection": {"enabled": True, "sensitivity": "high", "response_mode": "automatic", "alert_threshold": 0.7}
    }
  },
  "user_preferences": {
    "interface": {"theme": "dark", "language": "en-US", "voice_enabled": True, "notifications": True},
    "ai_behavior": {"interaction_style": "professional", "verbosity_level": "medium", "learning_rate": "adaptive", "personalization_enabled": True},
    "privacy": {"data_collection": "essential_only", "analytics_enabled": False, "telemetry_enabled": True, "sharing_permissions": []}
  }
}

def test_config_example_validates():
    cfg = validate_config(example)
    assert cfg.system_config.core.environment == "production"


def test_invalid_environment():
    bad = example.copy()
    bad["system_config"] = dict(bad["system_config"])  # shallow clone
    bad["system_config"]["core"] = dict(bad["system_config"]["core"])
    bad["system_config"]["core"]["environment"] = "invalid"
    try:
        validate_config(bad)
        assert False, "Should have raised"
    except Exception as e:
        assert "invalid environment" in str(e)
