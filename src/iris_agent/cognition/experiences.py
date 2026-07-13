from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from .types import Experience


class ExperienceStore:
    def __init__(self) -> None:
        self._experiences: Dict[str, Experience] = {}

    def create(
        self,
        goal: str,
        context: str,
        action: str,
        expected_outcome: str,
        actual_outcome: str,
        confidence: float = 0.0,
        reasoning_summary: str = "",
    ) -> Experience:
        exp = Experience(
            id=uuid.uuid4().hex[:12],
            goal=goal,
            context=context,
            action=action,
            expected_outcome=expected_outcome,
            actual_outcome=actual_outcome,
            confidence=confidence,
            timestamp=time.time(),
            reasoning_summary=reasoning_summary,
        )
        self._experiences[exp.id] = exp
        return exp

    def get(self, exp_id: str) -> Optional[Experience]:
        return self._experiences.get(exp_id)

    def add_lesson(self, exp_id: str, lesson: str) -> None:
        exp = self._experiences.get(exp_id)
        if exp:
            exp.lessons.append(lesson)

    def add_mistake(self, exp_id: str, mistake: str) -> None:
        exp = self._experiences.get(exp_id)
        if exp:
            exp.mistakes.append(mistake)

    def add_strategy(self, exp_id: str, strategy: str) -> None:
        exp = self._experiences.get(exp_id)
        if exp:
            exp.successful_strategies.append(strategy)

    def recent(self, limit: int = 10) -> List[Experience]:
        sorted_exps = sorted(
            self._experiences.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_exps[:limit]

    def all(self) -> List[Experience]:
        return list(self._experiences.values())

    def clear(self) -> None:
        self._experiences.clear()
