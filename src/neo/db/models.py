from __future__ import annotations
from sqlalchemy import Column, Integer, String, Boolean, DateTime, UniqueConstraint, ForeignKey, Float, JSON, Text
from sqlalchemy.sql import func
from neo.db import Base


class CognitiveRoadmapItem(Base):
    __tablename__ = "cognitive_roadmap_items"
    id = Column(Integer, primary_key=True)
    phase = Column(Integer, nullable=False, index=True)
    item = Column(String(255), nullable=False)
    implemented = Column(Boolean, default=False, nullable=False)
    description = Column(String(500), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint("phase", "item", name="u_phase_item"),)


class KnowledgeNodeModel(Base):
    __tablename__ = "knowledge_nodes"
    id = Column(String(64), primary_key=True)
    type = Column(String(100), index=True, nullable=False)
    properties = Column(JSON, nullable=False, default={})
    confidence_score = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KnowledgeEdgeModel(Base):
    __tablename__ = "knowledge_edges"
    id = Column(String(64), primary_key=True)
    source_id = Column(String(64), ForeignKey("knowledge_nodes.id", ondelete="CASCADE"), index=True, nullable=False)
    target_id = Column(String(64), ForeignKey("knowledge_nodes.id", ondelete="CASCADE"), index=True, nullable=False)
    relationship_type = Column(String(100), index=True, nullable=False)
    properties = Column(JSON, nullable=False, default={})
    weight = Column(Float, default=1.0, nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MemoryItemModel(Base):
    __tablename__ = "memory_items"
    id = Column(String(64), primary_key=True)
    kind = Column(String(40), index=True, nullable=False)  # episodic|semantic|procedural|working|short_term
    stage = Column(String(20), index=True, nullable=False)  # short_term|working|long_term
    content = Column(JSON, nullable=False, default={})
    context = Column(JSON, nullable=False, default={})
    encoding = Column(JSON, nullable=False, default={})
    importance = Column(JSON, nullable=False, default={})
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# --- Chat / Conversation Models ---
class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(64), primary_key=True)
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String(20), nullable=False)  # user|assistant|system
    content = Column(Text, nullable=False)
    meta = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
