from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from .auth import require_scope
from functools import lru_cache
from neo.datastructures.knowledge import KnowledgeGraph
from neo.db import get_session
from neo.services.knowledge_service import KnowledgeService
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

# Provide a singleton in-memory graph for now
@lru_cache(maxsize=1)
def get_graph() -> KnowledgeGraph:
    return KnowledgeGraph()

class NodeCreate(BaseModel):
    type: str
    properties: Dict[str, Any] = {}

class EdgeCreate(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = {}

class QueryParams(BaseModel):
    type: Optional[str] = None
    property_key: Optional[str] = None
    property_value: Optional[Any] = None

@router.post("/knowledge/nodes", dependencies=[Depends(require_scope("write:knowledge"))])
async def create_node(body: NodeCreate, backend: str = "memory", session: AsyncSession = Depends(get_session)):
    if backend == "persist":
        svc = KnowledgeService(session)
        node = await svc.create_node(body.type, body.properties)
        return {"id": node.id, "type": node.type, "properties": node.properties, "backend": "persist"}
    kg = get_graph()
    node = kg.add_node(body.type, body.properties)
    return {"id": node.id, "type": node.type, "properties": node.properties, "backend": "memory"}

@router.get("/knowledge/nodes/{node_id}", dependencies=[Depends(require_scope("read:knowledge"))])
async def get_node(node_id: str, backend: str = "memory", session: AsyncSession = Depends(get_session)):
    if backend == "persist":
        svc = KnowledgeService(session)
        node = await svc.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        return {"id": node.id, "type": node.type, "properties": node.properties, "backend": "persist"}
    kg = get_graph()
    node = kg.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"id": node.id, "type": node.type, "properties": node.properties, "relationships": node.relationships, "backend": "memory"}

@router.post("/knowledge/edges", dependencies=[Depends(require_scope("write:knowledge"))])
async def create_edge(body: EdgeCreate, backend: str = "memory", session: AsyncSession = Depends(get_session)):
    if backend == "persist":
        svc = KnowledgeService(session)
        edge = await svc.create_edge(body.source_id, body.target_id, body.relationship_type, body.properties)
        return {"id": edge.id, "source_id": edge.source_id, "target_id": edge.target_id, "relationship_type": edge.relationship_type, "backend": "persist"}
    kg = get_graph()
    edge = kg.add_edge(body.source_id, body.target_id, body.relationship_type, body.properties)
    return {"id": edge.source_id + ":" + edge.target_id, "source_id": edge.source_id, "target_id": edge.target_id, "relationship_type": edge.relationship_type, "backend": "memory"}

@router.get("/knowledge/query", dependencies=[Depends(require_scope("read:knowledge"))])
async def query_nodes(type: Optional[str] = None, property_key: Optional[str] = None, property_value: Optional[str] = None, backend: str = "memory", session: AsyncSession = Depends(get_session)):
    if backend == "persist":
        svc = KnowledgeService(session)
        nodes = await svc.query_nodes(type_ = type)
        return [{"id": n.id, "type": n.type, "properties": n.properties, "backend": "persist"} for n in nodes]
    kg = get_graph()
    if property_key and property_value:
        nodes = kg.query(type=type, property_eq=(property_key, property_value))
    elif type:
        nodes = kg.query(type=type)
    else:
        nodes = kg.query()
    return [{"id": n.id, "type": n.type, "properties": n.properties, "backend": "memory"} for n in nodes]

@router.get("/knowledge/nodes/{node_id}/neighbors", dependencies=[Depends(require_scope("read:knowledge"))])
async def neighbors(node_id: str, rel_type: Optional[str] = None, backend: str = "memory", session: AsyncSession = Depends(get_session)):
    if backend == "persist":
        svc = KnowledgeService(session)
        if not await svc.get_node(node_id):
            raise HTTPException(status_code=404, detail="Node not found")
        nodes = await svc.neighbors(node_id)
        return [{"id": n.id, "type": n.type, "properties": n.properties, "backend": "persist"} for n in nodes]
    kg = get_graph()
    if not kg.get_node(node_id):
        raise HTTPException(status_code=404, detail="Node not found")
    return [{"id": n.id, "type": n.type, "properties": n.properties, "backend": "memory"} for n in kg.neighbors(node_id, rel_type=rel_type)]

@router.delete("/knowledge/nodes/{node_id}", dependencies=[Depends(require_scope("write:knowledge"))])
async def delete_node(node_id: str, backend: str = "memory", session: AsyncSession = Depends(get_session)):
    if backend == "persist":
        svc = KnowledgeService(session)
        ok = await svc.delete_node(node_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Node not found")
        return {"deleted": node_id, "backend": "persist"}
    kg = get_graph()
    if not kg.get_node(node_id):
        raise HTTPException(status_code=404, detail="Node not found")
    kg.remove_node(node_id)
    return {"deleted": node_id, "backend": "memory"}

@router.get("/knowledge/traverse/{node_id}", dependencies=[Depends(require_scope("read:knowledge"))])
async def traverse(node_id: str, max_depth: int = 3, rel_type: Optional[str] = None):
    kg = get_graph()
    if not kg.get_node(node_id):
        raise HTTPException(status_code=404, detail="Node not found")
    nodes = kg.traverse_bfs(node_id, max_depth=max_depth, rel_type=rel_type)
    return [{"id": n.id, "type": n.type, "properties": n.properties} for n in nodes]
