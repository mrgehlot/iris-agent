from __future__ import annotations

from typing import Any, Dict, List

from .base import CognitiveModule
from .events import CognitiveEventType
from .types import Critique


class Critic(CognitiveModule):
    def __init__(self, min_score: float = 0.5) -> None:
        super().__init__()
        self._min_score = min_score

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        plan = context.get("plan", [])
        if not plan:
            return context

        issues: List[str] = []
        suggestions: List[str] = []

        for i, step in enumerate(plan):
            action = step.get("action", "")
            if not action:
                issues.append(f"Step {i+1} has no action defined")
                suggestions.append(f"Define a concrete action for step {i+1}")

        score = 1.0 - (len(issues) * 0.2)
        score = max(0.0, min(1.0, score))

        critique = Critique(
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

        context["plan_critique"] = f"Score: {score:.2f}\nIssues: {', '.join(issues) if issues else 'None'}\nSuggestions: {', '.join(suggestions) if suggestions else 'None'}"
        context["plan_score"] = score

        if self.mind:
            self.mind.event_bus.emit(
                CognitiveEventType.PLAN_CRITIQUED,
                score=score,
                issues=issues,
                suggestions=suggestions,
            )

        if score < self._min_score:
            context["plan_revision_needed"] = True
            context["plan_revision_reason"] = f"Score {score:.2f} below minimum {self._min_score}"
        else:
            context["plan_revision_needed"] = False

        return context
