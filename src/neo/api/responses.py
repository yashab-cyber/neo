"""Response helpers."""
from __future__ import annotations
from typing import Any


def error_response(code: str, message: str, details: Any, request_id: str | None):
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": request_id,
        }
    }
