from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .base import CognitiveModule
from .types import Hypothesis


class HypothesisGenerator(CognitiveModule):
    def __init__(self) -> None:
        super().__init__()
        self._hypotheses: Dict[str, Hypothesis] = {}

    def propose(self, statement: str, predictions: List[str] | None = None) -> Hypothesis:
        h = Hypothesis(
            id=uuid.uuid4().hex[:12],
            statement=statement,
            predictions=predictions or [],
        )
        self._hypotheses[h.id] = h
        return h

    def test(self, hypothesis_id: str, observation: str, supports: bool) -> Optional[bool]:
        h = self._hypotheses.get(hypothesis_id)
        if not h:
            return None
        h.evidence.append(observation)
        if supports:
            h.confidence = min(1.0, h.confidence + 0.1)
        else:
            h.confidence = max(0.0, h.confidence - 0.15)
        return supports

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return self._hypotheses.get(hypothesis_id)

    def active(self) -> List[Hypothesis]:
        return [h for h in self._hypotheses.values() if h.status == "proposed"]

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context
