from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from .types import Decision


class DecisionLog:
    def __init__(self) -> None:
        self._decisions: Dict[str, Decision] = {}

    def record(
        self,
        goal: str,
        options: List[str],
        chosen: str,
        reasoning: str,
        confidence: float = 0.0,
    ) -> Decision:
        decision = Decision(
            id=uuid.uuid4().hex[:12],
            goal=goal,
            options=options,
            chosen=chosen,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=time.time(),
        )
        self._decisions[decision.id] = decision
        return decision

    def get(self, decision_id: str) -> Optional[Decision]:
        return self._decisions.get(decision_id)

    def recent(self, limit: int = 10) -> List[Decision]:
        sorted_decisions = sorted(
            self._decisions.values(),
            key=lambda d: d.timestamp,
            reverse=True,
        )
        return sorted_decisions[:limit]

    def all(self) -> List[Decision]:
        return list(self._decisions.values())

    def clear(self) -> None:
        self._decisions.clear()
