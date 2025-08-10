from __future__ import annotations
from sqlalchemy import Column, Integer, String, Boolean, DateTime, UniqueConstraint, ForeignKey, Float, JSON
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
