from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from .types import Belief


class BeliefSystem:
    def __init__(self) -> None:
        self._beliefs: Dict[str, Belief] = {}

    def get_or_create(self, statement: str) -> Belief:
        for belief in self._beliefs.values():
            if belief.statement == statement:
                return belief
        belief = Belief(
            id=uuid.uuid4().hex[:12],
            statement=statement,
            confidence=0.5,
            last_updated=time.time(),
        )
        self._beliefs[belief.id] = belief
        return belief

    def update(
        self,
        statement: str,
        evidence: str,
        supports: bool,
        evidence_weight: float = 0.1,
    ) -> Belief:
        belief = self.get_or_create(statement)
        if supports:
            belief.supporting_evidence.append(evidence)
            belief.confidence = min(
                1.0, belief.confidence + evidence_weight * (1.0 - belief.confidence)
            )
        else:
            belief.contradicting_evidence.append(evidence)
            belief.confidence = max(
                0.0, belief.confidence - evidence_weight * belief.confidence
            )
        belief.last_updated = time.time()
        return belief

    def get(self, belief_id: str) -> Optional[Belief]:
        return self._beliefs.get(belief_id)

    def find(self, statement: str) -> Optional[Belief]:
        for belief in self._beliefs.values():
            if belief.statement == statement:
                return belief
        return None

    def all(self) -> List[Belief]:
        return list(self._beliefs.values())

    def confident(self, threshold: float = 0.7) -> List[Belief]:
        return [b for b in self._beliefs.values() if b.confidence >= threshold]

    def uncertain(self, threshold: float = 0.5) -> List[Belief]:
        return [b for b in self._beliefs.values() if b.confidence < threshold]

    def remove(self, belief_id: str) -> None:
        self._beliefs.pop(belief_id, None)

    def clear(self) -> None:
        self._beliefs.clear()
