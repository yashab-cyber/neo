from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from .auth import require_scope
from functools import lru_cache
from neo.datastructures.knowledge import KnowledgeGraph
from neo.db import get_session
from neo.services.knowledge_service import KnowledgeService
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from neo.db.models import KnowledgeNodeModel, KnowledgeEdgeModel

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


@router.get("/knowledge/status", dependencies=[Depends(require_scope("read:knowledge"))])
async def knowledge_status(session: AsyncSession = Depends(get_session)):
    """Return implementation status / metrics for knowledge graph subsystems.

    Heuristic detection based on current data & capabilities:
    - entity_layer: memory graph has representative entity types (Person, Organization, Concept, Event, Object, Location)
    - relationship_layer: at least one edge in either backend
    - attribute_layer: at least one node with properties in either backend
    - domain_knowledge: presence of domain category nodes (Technology, Science, Business, Security, Healthcare, Education)
    Remaining layers presently placeholders (False) until implemented.
    """
    kg = get_graph()
    mem_nodes = list(kg.nodes.values())
    mem_edges = list(kg.edges.values())
    # persistent counts
    node_count_persist = (await session.execute(select(func.count()).select_from(KnowledgeNodeModel))).scalar() or 0
    edge_count_persist = (await session.execute(select(func.count()).select_from(KnowledgeEdgeModel))).scalar() or 0

    # detection helpers
    required_entities = {"Person", "Organization", "Concept", "Event", "Object", "Location"}
    mem_entity_types = {n.type for n in mem_nodes}
    entity_layer_ok = required_entities.issubset(mem_entity_types)

    relationship_layer_ok = bool(mem_edges or edge_count_persist > 0)
    attribute_layer_ok = any(n.properties for n in mem_nodes) or node_count_persist > 0
    domain_required = {"Technology", "Science", "Business", "Security", "Healthcare", "Education"}
    domain_ok = domain_required.intersection(mem_entity_types)
    domain_layer_ok = len(domain_ok) >= 3  # partial for now

    layers = [
        {"layer": "entity_layer", "implemented": entity_layer_ok, "required_types": sorted(required_entities)},
        {"layer": "relationship_layer", "implemented": relationship_layer_ok},
        {"layer": "attribute_layer", "implemented": attribute_layer_ok},
        {"layer": "domain_knowledge", "implemented": domain_layer_ok, "detected": sorted(domain_ok)},
        {"layer": "inference_engine", "implemented": False},
        {"layer": "acquisition_pipeline", "implemented": False},
        {"layer": "query_reasoning", "implemented": False},
        {"layer": "domain_models", "implemented": False},
        {"layer": "evolution_versioning", "implemented": False},
    ]
    implemented = sum(1 for l in layers if l["implemented"])
    total = len(layers)
    return {
        "layers": layers,
        "implemented": implemented,
        "total": total,
        "completion": implemented / total if total else 0.0,
        "memory": {"nodes": len(mem_nodes), "edges": len(mem_edges)},
        "persistent": {"nodes": node_count_persist, "edges": edge_count_persist},
    }


class LearningLayerStatus(BaseModel):
    name: str
    implemented: bool
    description: str


class LearningStatusResponse(BaseModel):
    layers: List[LearningLayerStatus]
    completion: float
    metrics: Dict[str, Any]


@router.get("/knowledge/learning/status", dependencies=[Depends(require_scope("read:knowledge"))])
async def learning_status():
    """Report heuristic implementation status for learning data structures derived from config.

    Uses presence of expected keys in config_example.yaml under learning_data.* to mark a layer implemented.
    Future: replace with dynamic runtime services & metrics.
    """
    layer_specs: Dict[str, Dict[str, Any]] = {
        "input_data": {
            "description": "Raw, preprocessed, feature, embedding, sequence & structured inputs",
            "paths": [
                "learning_data.input_data.raw",
                "learning_data.input_data.preprocessed",
                "learning_data.input_data.features",
            ],
        },
        "deep_learning_structures": {
            "description": "Tensors, transformer blocks, weights, gradients, activations, attention",
            "paths": [
                "learning_data.deep.tensors",
                "learning_data.deep.transformer_blocks",
            ],
        },
        "neuro_learning_structures": {
            "description": "Synapses, plasticity, spikes, memory traces, neural states",
            "paths": [
                "learning_data.neuro.synapses",
                "learning_data.neuro.plasticity",
            ],
        },
        "recursive_learning_structures": {
            "description": "Policies, meta params, experience, adaptation histories",
            "paths": [
                "learning_data.recursive.policies",
                "learning_data.recursive.meta_params",
            ],
        },
        "storage_management": {
            "description": "Versioning, checkpoints, compression, sharding, caching lifecycle",
            "paths": [
                "learning_data.storage.versioning",
                "learning_data.storage.checkpoints",
            ],
        },
    }

    # Load config file (example). In a full system this would come from active runtime config service.
    from pathlib import Path
    import yaml  # type: ignore

    config_path = Path(__file__).resolve().parent.parent.parent / "cognitive" / "config_example.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}

    def fetch(path: str):
        cur: Any = cfg
        for part in path.split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    layers: List[LearningLayerStatus] = []
    implemented_count = 0
    for name, spec in layer_specs.items():
        present = all(fetch(p) not in (None, False) for p in spec["paths"]) if spec["paths"] else False
        if present:
            implemented_count += 1
        layers.append(LearningLayerStatus(name=name, implemented=present, description=spec["description"]))

    total = len(layer_specs)
    completion = implemented_count / total if total else 0.0
    metrics = {
        "layers_total": total,
        "layers_implemented": implemented_count,
        "config_loaded": bool(cfg),
    }
    return LearningStatusResponse(layers=layers, completion=completion, metrics=metrics)
