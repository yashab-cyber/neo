from __future__ import annotations
import uuid, random
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from neo.db.models import ChatSessionModel, ChatMessageModel


class ChatService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_session(self, title: str | None = None) -> ChatSessionModel:
        sid = uuid.uuid4().hex[:32]
        obj = ChatSessionModel(id=sid, title=title)
        self.session.add(obj)
        await self.session.commit()
        return obj

    async def get_session(self, session_id: str) -> Optional[ChatSessionModel]:
        return await self.session.get(ChatSessionModel, session_id)

    async def list_sessions(self, limit: int = 20) -> List[ChatSessionModel]:
        res = await self.session.execute(select(ChatSessionModel).order_by(ChatSessionModel.updated_at.desc()).limit(limit))
        return list(res.scalars())

    async def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] | None = None) -> ChatMessageModel:
        mid = uuid.uuid4().hex[:32]
        msg = ChatMessageModel(id=mid, session_id=session_id, role=role, content=content, meta=metadata or {})
        self.session.add(msg)
        await self.session.commit()
        return msg

    async def get_messages(self, session_id: str, limit: int = 50) -> List[ChatMessageModel]:
        res = await self.session.execute(select(ChatMessageModel).where(ChatMessageModel.session_id == session_id).order_by(ChatMessageModel.created_at.asc()).limit(limit))
        return list(res.scalars())

    async def chat(self, message: str, session_id: str | None = None, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if session_id:
            sess = await self.get_session(session_id)
            if not sess:
                sess = await self.create_session()
                session_id = sess.id
        else:
            sess = await self.create_session()
            session_id = sess.id
        await self.add_message(session_id, "user", message, metadata={"context": context or {}})
        history = await self.get_messages(session_id, limit=20)
        keywords = [w for w in message.split() if len(w) > 4][:5]
        reply_core = "I registered your message. "
        if keywords:
            reply_core += "Key terms: " + ", ".join(keywords) + ". "
        if context:
            reply_core += f"Context keys: {', '.join(context.keys())}. "
        reply = reply_core + "(This is a placeholder model response.)"
        await self.add_message(session_id, "assistant", reply, metadata={"keywords": keywords})
        return {
            "session_id": session_id,
            "response": reply,
            "confidence": round(random.uniform(0.75, 0.9), 2),
            "tokens_used": len(message.split()) * 2 + sum(len(m.content.split()) for m in history[-3:]),
            "history_size": len(history) + 1,
            "suggestions": [
                {"action": "store_memory", "command": "memory.store", "confidence": 0.6},
            ],
        }

__all__ = ["ChatService"]
