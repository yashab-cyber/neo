from __future__ import annotations
from typing import Optional, Dict, Any, List
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from neo.db.models import KnowledgeNodeModel, KnowledgeEdgeModel


class KnowledgeService:
    """Persistence-backed knowledge graph service (nodes then edges)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # Nodes -----------------------------------------------------------------
    async def create_node(self, type_: str, properties: Dict[str, Any]) -> KnowledgeNodeModel:
        node_id = uuid.uuid4().hex[:32]
        node = KnowledgeNodeModel(id=node_id, type=type_, properties=properties or {})
        self.session.add(node)
        await self.session.commit()
        return node

    async def get_node(self, node_id: str) -> Optional[KnowledgeNodeModel]:
        return await self.session.get(KnowledgeNodeModel, node_id)

    async def query_nodes(self, type_: Optional[str] = None) -> List[KnowledgeNodeModel]:
        stmt = select(KnowledgeNodeModel)
        if type_:
            stmt = stmt.where(KnowledgeNodeModel.type == type_)
        res = await self.session.execute(stmt)
        return list(res.scalars())

    # Edges -----------------------------------------------------------------
    async def create_edge(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any]) -> KnowledgeEdgeModel:
        # ensure both nodes exist
        if not await self.get_node(source_id) or not await self.get_node(target_id):
            raise ValueError("Source or target node does not exist")
        edge_id = uuid.uuid4().hex[:32]
        edge = KnowledgeEdgeModel(id=edge_id, source_id=source_id, target_id=target_id, relationship_type=rel_type, properties=properties or {})
        self.session.add(edge)
        await self.session.commit()
        return edge

    async def neighbors(self, node_id: str) -> List[KnowledgeNodeModel]:
        # naive: fetch edges where source=node
        stmt = select(KnowledgeEdgeModel).where(KnowledgeEdgeModel.source_id == node_id)
        res = await self.session.execute(stmt)
        edges = list(res.scalars())
        if not edges:
            return []
        target_ids = [e.target_id for e in edges]
        stmt2 = select(KnowledgeNodeModel).where(KnowledgeNodeModel.id.in_(target_ids))
        res2 = await self.session.execute(stmt2)
        return list(res2.scalars())

    async def delete_node(self, node_id: str) -> bool:
        # cascade edges manually (db ondelete may not fire with sqlite constraints -- enforce)
        await self.session.execute(delete(KnowledgeEdgeModel).where((KnowledgeEdgeModel.source_id == node_id) | (KnowledgeEdgeModel.target_id == node_id)))
        res = await self.session.execute(delete(KnowledgeNodeModel).where(KnowledgeNodeModel.id == node_id))
        await self.session.commit()
        return res.rowcount > 0

__all__ = ["KnowledgeService"]
