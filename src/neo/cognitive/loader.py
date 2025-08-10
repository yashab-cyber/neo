from __future__ import annotations
from pathlib import Path
import yaml
from functools import lru_cache
from typing import Any, Dict

DEFAULT_PATH = Path(__file__).with_name("config_example.yaml")


@lru_cache
def _load_raw(path: str | None = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():  # pragma: no cover
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_cognitive_settings(path: str | None = None) -> Dict[str, Any]:
    return _load_raw(path)


cognitive_settings = get_cognitive_settings()
