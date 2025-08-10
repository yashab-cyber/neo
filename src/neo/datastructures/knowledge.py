from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Any, Iterable, Optional
import uuid


@dataclass
class KnowledgeNode:
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # relationship_type -> list[node_id]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence_score: float = 1.0

    def add_relationship(self, rel_type: str, target_id: str):
        self.relationships.setdefault(rel_type, [])
        if target_id not in self.relationships[rel_type]:
            self.relationships[rel_type].append(target_id)
            self.updated_at = datetime.now(UTC)


@dataclass
class KnowledgeEdge:
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class KnowledgeGraph:
    """In-memory knowledge graph with simple indexing & traversal utilities."""
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        # simple indexes
        self.type_index: Dict[str, List[str]] = {}
        self.property_index: Dict[str, Dict[Any, List[str]]] = {}
        self.relationship_index: Dict[str, List[str]] = {}  # rel_type -> edge_ids

    # Node operations -------------------------------------------------
    def add_node(self, node_type: str, properties: Optional[Dict[str, Any]] = None, *, node_id: Optional[str] = None) -> KnowledgeNode:
        node_id = node_id or str(uuid.uuid4())
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        node = KnowledgeNode(id=node_id, type=node_type, properties=properties or {})
        self.nodes[node_id] = node
        self.type_index.setdefault(node_type, []).append(node_id)
        for k, v in node.properties.items():
            self.property_index.setdefault(k, {}).setdefault(v, []).append(node_id)
        return node

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        return self.nodes.get(node_id)

    def find_by_type(self, node_type: str) -> List[KnowledgeNode]:
        return [self.nodes[nid] for nid in self.type_index.get(node_type, [])]

    def find_by_property(self, key: str, value: Any) -> List[KnowledgeNode]:
        return [self.nodes[nid] for nid in self.property_index.get(key, {}).get(value, [])]

    # Edge operations -------------------------------------------------
    def add_edge(self, source_id: str, target_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> KnowledgeEdge:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist")
        edge_id = str(uuid.uuid4())
        edge = KnowledgeEdge(source_id=source_id, target_id=target_id, relationship_type=relationship_type, properties=properties or {})
        self.edges[edge_id] = edge
        self.nodes[source_id].add_relationship(relationship_type, target_id)
        self.relationship_index.setdefault(relationship_type, []).append(edge_id)
        return edge

    def neighbors(self, node_id: str, rel_type: Optional[str] = None) -> Iterable[KnowledgeNode]:
        node = self.get_node(node_id)
        if not node:
            return []
        if rel_type:
            ids = node.relationships.get(rel_type, [])
        else:
            # flatten all relationships
            ids = [tid for lst in node.relationships.values() for tid in lst]
        for nid in ids:
            n = self.nodes.get(nid)
            if n:
                yield n

    # Traversal -------------------------------------------------------
    def traverse_bfs(self, start_id: str, max_depth: int = 3, rel_type: Optional[str] = None) -> List[KnowledgeNode]:
        if start_id not in self.nodes:
            return []
        visited = {start_id}
        queue = [(start_id, 0)]
        result = []
        while queue:
            nid, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for nb in self.neighbors(nid, rel_type=rel_type):
                if nb.id not in visited:
                    visited.add(nb.id)
                    result.append(nb)
                    queue.append((nb.id, depth + 1))
        return result

    # Query pattern (simple) -----------------------------------------
    def query(self, *, type: Optional[str] = None, property_eq: Optional[tuple] = None) -> List[KnowledgeNode]:
        if type and property_eq:
            key, value = property_eq
            ids_type = set(self.type_index.get(type, []))
            ids_prop = set(self.property_index.get(key, {}).get(value, []))
            return [self.nodes[i] for i in ids_type & ids_prop]
        if type:
            return self.find_by_type(type)
        if property_eq:
            key, value = property_eq
            return self.find_by_property(key, value)
        return list(self.nodes.values())

    # Maintenance -----------------------------------------------------
    def remove_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if not node:
            return
        # remove from type index
        if node.type in self.type_index:
            self.type_index[node.type] = [nid for nid in self.type_index[node.type] if nid != node_id]
            if not self.type_index[node.type]:
                del self.type_index[node.type]
        # remove from property index
        for k, v in node.properties.items():
            bucket = self.property_index.get(k, {}).get(v, [])
            self.property_index[k][v] = [nid for nid in bucket if nid != node_id]
            if not self.property_index[k][v]:
                del self.property_index[k][v]
            if not self.property_index[k]:
                del self.property_index[k]
        # remove edges referencing node
        edges_to_remove = [eid for eid, e in self.edges.items() if e.source_id == node_id or e.target_id == node_id]
        for eid in edges_to_remove:
            self.remove_edge(eid)

    def remove_edge(self, edge_id: str) -> None:
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return
        # prune relationship index
        rel_list = self.relationship_index.get(edge.relationship_type, [])
        self.relationship_index[edge.relationship_type] = [eid for eid in rel_list if eid != edge_id]
        if not self.relationship_index[edge.relationship_type]:
            del self.relationship_index[edge.relationship_type]
        # prune relationship from source node
        src = self.nodes.get(edge.source_id)
        if src:
            lst = src.relationships.get(edge.relationship_type, [])
            if edge.target_id in lst:
                lst.remove(edge.target_id)
                if not lst:
                    del src.relationships[edge.relationship_type]
                src.updated_at = datetime.now(UTC)


__all__ = ["KnowledgeNode", "KnowledgeEdge", "KnowledgeGraph"]
