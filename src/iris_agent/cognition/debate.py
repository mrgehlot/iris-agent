from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import CognitiveModule


@dataclass
class DebateRole:
    name: str
    perspective: str
    prompt_instructions: str = ""


@dataclass
class DebateViewpoint:
    role: str
    argument: str
    confidence: float = 0.5


@dataclass
class DebateResult:
    question: str
    viewpoints: List[DebateViewpoint]
    consensus: str = ""
    dissenting: List[str] = field(default_factory=list)


DEFAULT_ROLES = [
    DebateRole("engineer", "Focus on technical feasibility and implementation details."),
    DebateRole("architect", "Focus on system design, scalability, and trade-offs."),
    DebateRole("security", "Focus on security implications, vulnerabilities, and risk mitigation."),
    DebateRole("product_manager", "Focus on user value, priorities, and business goals."),
    DebateRole("critic", "Focus on identifying flaws, assumptions, and potential failures."),
    DebateRole("economist", "Focus on cost-effectiveness, resource allocation, and ROI."),
]


class DebateEngine(CognitiveModule):
    def __init__(self, roles: List[DebateRole] | None = None) -> None:
        super().__init__()
        self._roles = roles or DEFAULT_ROLES

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        question = context.get("goal", context.get("input", ""))
        if not question:
            return context

        viewpoints = self._gather_viewpoints(question, context)
        consensus, dissenting = self._aggregate(viewpoints)

        result = DebateResult(
            question=question,
            viewpoints=viewpoints,
            consensus=consensus,
            dissenting=dissenting,
        )

        context["debate_result"] = result
        context["debate_consensus"] = consensus
        context["debate_dissenting"] = dissenting

        return context

    def _gather_viewpoints(self, question: str, context: Dict[str, Any]) -> List[DebateViewpoint]:
        return [
            DebateViewpoint(
                role=role.name,
                argument=f"[{role.name} perspective on: {question}]\n{role.perspective}",
                confidence=0.7,
            )
            for role in self._roles
        ]

    def _aggregate(self, viewpoints: List[DebateViewpoint]) -> tuple:
        if not viewpoints:
            return "", []
        consensus = viewpoints[0].argument
        dissenting = [v.argument for v in viewpoints[1:]]
        return consensus, dissenting
