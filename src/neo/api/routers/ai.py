from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List
import random, time

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] | None = None
    model: str | None = "stub-model"
    stream: bool | None = False


class ChatResponse(BaseModel):
    response: str
    confidence: float
    tokens_used: int
    processing_time: float
    suggestions: List[Dict[str, Any]] | None


@router.post("/ai/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    # Placeholder logic
    suggestions = [
        {"action": "optimize_memory", "confidence": 0.75, "command": "system.optimize_memory"}
    ]
    return ChatResponse(
        response=f"Stub response to: {req.message}",
        confidence=round(random.uniform(0.7, 0.95), 2),
        tokens_used=len(req.message.split()) * 2,
        processing_time=round(time.time() - start, 3),
        suggestions=suggestions,
    )
