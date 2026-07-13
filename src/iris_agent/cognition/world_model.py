from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .graph import Entity, KnowledgeGraph


class WorldModel:
    def __init__(self) -> None:
        self.graph = KnowledgeGraph()
        self._current_goal_id: str = ""
        self._session_id: str = uuid.uuid4().hex[:12]

    @property
    def current_goal_id(self) -> str:
        return self._current_goal_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def ingest_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        props = {"role": role, "content": content}
        if metadata:
            props.update(metadata)
        return self.graph.add_entity("message", props)

    def set_goal(
        self, goal: str, properties: Optional[Dict[str, Any]] = None
    ) -> str:
        props = {"description": goal}
        if properties:
            props.update(properties)
        goal_id = self.graph.add_entity("goal", props)
        self._current_goal_id = goal_id
        return goal_id

    def add_task(
        self,
        task: str,
        goal_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        props = {"description": task}
        if properties:
            props.update(properties)
        task_id = self.graph.add_entity("task", props)
        parent_goal = goal_id or self._current_goal_id
        if parent_goal:
            self.graph.add_relation(task_id, parent_goal, "subgoal_of")
        return task_id

    def add_entity(
        self,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        entity_id: str = "",
    ) -> str:
        return self.graph.add_entity(entity_type, properties, entity_id)

    def link(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.graph.add_relation(source_id, target_id, rel_type, properties)

    def get_related(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Entity]:
        return self.graph.get_related(entity_id, rel_type, direction)

    def get_context(self, max_entities: int = 15) -> str:
        parts: List[str] = []

        if self._current_goal_id:
            goal = self.graph.get_entity(self._current_goal_id)
            if goal:
                parts.append(f"Current Goal: {goal.properties.get('description', '')}")

        entities = self.graph.find_entities()
        if entities:
            parts.append("\nKnown entities:")
            for e in entities[:max_entities]:
                desc = (
                    e.properties.get("description")
                    or e.properties.get("content", "")
                )
                snippet = desc[:100].replace("\n", " ")
                parts.append(f"  - [{e.type}] {snippet}")

        relations = self.graph.get_relations()
        if relations:
            parts.append("\nRelationships:")
            for r in relations[:max_entities]:
                src = self.graph.get_entity(r.source_id)
                tgt = self.graph.get_entity(r.target_id)
                src_name = (
                    src.properties.get("description", src.id[:8])
                    if src
                    else r.source_id[:8]
                )
                tgt_name = (
                    tgt.properties.get("description", tgt.id[:8])
                    if tgt
                    else r.target_id[:8]
                )
                parts.append(f"  - {src_name} --[{r.type}]--> {tgt_name}")

        return "\n".join(parts) if parts else ""

    def ingest_tool_result(
        self,
        tool_name: str,
        arguments: str,
        result: str,
    ) -> str:
        return self.graph.add_entity(
            "tool_result",
            {"tool": tool_name, "arguments": arguments, "result": result},
        )

    def record_belief(self, statement: str, confidence: float = 0.5) -> str:
        eid = self.graph.add_entity(
            "belief",
            {"statement": statement, "confidence": confidence},
        )
        if self._current_goal_id:
            self.graph.add_relation(eid, self._current_goal_id, "relevant_to")
        return eid

    def record_lesson(self, lesson: str) -> str:
        eid = self.graph.add_entity("lesson", {"description": lesson})
        if self._current_goal_id:
            self.graph.add_relation(eid, self._current_goal_id, "learned_from")
        return eid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self._session_id,
            "current_goal_id": self._current_goal_id,
            "graph": self.graph.to_dict(),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModel":
        wm = cls()
        wm._session_id = data.get("session_id", uuid.uuid4().hex[:12])
        wm._current_goal_id = data.get("current_goal_id", "")
        wm.graph = KnowledgeGraph.from_dict(data.get("graph", {}))
        return wm
