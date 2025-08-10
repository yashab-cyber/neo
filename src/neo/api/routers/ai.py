from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import time
from sqlalchemy.ext.asyncio import AsyncSession
from neo.db import get_session
from neo.services.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] | None = None
    model: str | None = "stub-model"
    stream: bool | None = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    confidence: float
    tokens_used: int
    processing_time: float
    history_size: int
    suggestions: List[Dict[str, Any]] | None


@router.post("/ai/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, session: AsyncSession = Depends(get_session)):
    start = time.time()
    svc = ChatService(session)
    result = await svc.chat(req.message, session_id=req.session_id, context=req.context or {})
    return ChatResponse(
        session_id=result["session_id"],
        response=result["response"],
        confidence=result["confidence"],
        tokens_used=result["tokens_used"],
        processing_time=round(time.time() - start, 3),
        history_size=result["history_size"],
        suggestions=result["suggestions"],
    )
