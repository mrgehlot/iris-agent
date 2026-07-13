from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Phase 1 — Cognition graph (kept for backward compat)
# ---------------------------------------------------------------------------


@dataclass
class CognitionNode:
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class CognitionEdge:
    source_id: str
    target_id: str
    relation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphStore(ABC):
    @abstractmethod
    def add_node(self, node: CognitionNode) -> None:
        ...

    @abstractmethod
    def add_edge(self, edge: CognitionEdge) -> None:
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[CognitionNode]:
        ...

    @abstractmethod
    def get_edges(self, node_id: str) -> List[CognitionEdge]:
        ...

    @abstractmethod
    def get_nodes_by_type(self, node_type: str) -> List[CognitionNode]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def all_nodes(self) -> List[CognitionNode]:
        ...

    @abstractmethod
    def all_edges(self) -> List[CognitionEdge]:
        ...


class DictGraphStore(GraphStore):
    def __init__(self) -> None:
        self._nodes: Dict[str, CognitionNode] = {}
        self._edges: List[CognitionEdge] = []

    def add_node(self, node: CognitionNode) -> None:
        if not node.id:
            node.id = uuid.uuid4().hex[:12]
        if not node.timestamp:
            node.timestamp = time.time()
        self._nodes[node.id] = node

    def add_edge(self, edge: CognitionEdge) -> None:
        self._edges.append(edge)

    def get_node(self, node_id: str) -> Optional[CognitionNode]:
        return self._nodes.get(node_id)

    def get_edges(self, node_id: str) -> List[CognitionEdge]:
        return [
            e
            for e in self._edges
            if e.source_id == node_id or e.target_id == node_id
        ]

    def get_nodes_by_type(self, node_type: str) -> List[CognitionNode]:
        return [n for n in self._nodes.values() if n.type == node_type]

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    def all_nodes(self) -> List[CognitionNode]:
        return list(self._nodes.values())

    def all_edges(self) -> List[CognitionEdge]:
        return list(self._edges)


# ---------------------------------------------------------------------------
# Phase 2 — Knowledge Graph (World Model)
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class Relation:
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


class KnowledgeGraph:
    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}

    # ---- entities ----

    def add_entity(
        self,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        entity_id: str = "",
    ) -> str:
        eid = entity_id or uuid.uuid4().hex[:12]
        now = time.time()
        entity = Entity(
            id=eid,
            type=entity_type,
            properties=properties or {},
            created_at=now,
            updated_at=now,
        )
        self._entities[eid] = entity
        return eid

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        entity = self._entities.get(entity_id)
        if entity is None:
            return False
        entity.properties.update(properties)
        entity.updated_at = time.time()
        return True

    def delete_entity(self, entity_id: str) -> bool:
        if entity_id in self._entities:
            del self._entities[entity_id]
            self._relations = {
                rid: r
                for rid, r in self._relations.items()
                if r.source_id != entity_id and r.target_id != entity_id
            }
            return True
        return False

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        results = list(self._entities.values())
        if entity_type:
            results = [e for e in results if e.type == entity_type]
        if query:
            for key, value in query.items():
                results = [e for e in results if e.properties.get(key) == value]
        return results

    # ---- relations ----

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        rid = uuid.uuid4().hex[:12]
        relation = Relation(
            id=rid,
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            properties=properties or {},
            created_at=time.time(),
        )
        self._relations[rid] = relation
        return rid

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        return self._relations.get(relation_id)

    def get_relations(
        self,
        source_id: Optional[str] = None,
        rel_type: Optional[str] = None,
    ) -> List[Relation]:
        results = list(self._relations.values())
        if source_id:
            results = [r for r in results if r.source_id == source_id]
        if rel_type:
            results = [r for r in results if r.type == rel_type]
        return results

    def delete_relation(self, relation_id: str) -> bool:
        if relation_id in self._relations:
            del self._relations[relation_id]
            return True
        return False

    # ---- traversal ----

    def get_related(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Entity]:
        related_ids: set = set()
        for r in self._relations.values():
            if rel_type and r.type != rel_type:
                continue
            if direction in ("outgoing", "both") and r.source_id == entity_id:
                related_ids.add(r.target_id)
            if direction in ("incoming", "both") and r.target_id == entity_id:
                related_ids.add(r.source_id)
        return [self._entities[eid] for eid in related_ids if eid in self._entities]

    def traverse(
        self,
        start_id: str,
        rel_types: Optional[List[str]] = None,
        max_depth: int = 3,
    ) -> List[Entity]:
        visited: set = set()
        result: List[Entity] = []

        def _walk(current_id: str, depth: int) -> None:
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)
            entity = self._entities.get(current_id)
            if entity:
                result.append(entity)
            for r in self._relations.values():
                if rel_types and r.type not in rel_types:
                    continue
                if r.source_id == current_id:
                    _walk(r.target_id, depth + 1)
                elif r.target_id == current_id:
                    _walk(r.source_id, depth + 1)

        _walk(start_id, 0)
        return result

    # ---- bulk operations ----

    def clear(self) -> None:
        self._entities.clear()
        self._relations.clear()

    def stats(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        for e in self._entities.values():
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
        rel_counts: Dict[str, int] = {}
        for r in self._relations.values():
            rel_counts[r.type] = rel_counts.get(r.type, 0) + 1
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entity_types": type_counts,
            "relation_types": rel_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {
                eid: {
                    "id": e.id,
                    "type": e.type,
                    "properties": e.properties,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                }
                for eid, e in self._entities.items()
            },
            "relations": {
                rid: {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.type,
                    "properties": r.properties,
                    "created_at": r.created_at,
                }
                for rid, r in self._relations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        kg = cls()
        for eid, e_data in data.get("entities", {}).items():
            kg._entities[eid] = Entity(**e_data)
        for rid, r_data in data.get("relations", {}).items():
            kg._relations[rid] = Relation(**r_data)
        return kg
