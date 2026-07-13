from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MentalModelResult:
    model_name: str
    reasoning: str
    explanation: str
    confidence: float = 0.5
    suggestions: List[str] = field(default_factory=list)


class MentalModel(ABC):
    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def applicability(self, context: Dict[str, Any]) -> float:
        ...

    @abstractmethod
    def reason(self, context: Dict[str, Any]) -> MentalModelResult:
        ...


class FirstPrinciples(MentalModel):
    def __init__(self) -> None:
        super().__init__(
            name="first_principles",
            description="Break down complex problems into their most fundamental truths.",
        )

    def applicability(self, context: Dict[str, Any]) -> float:
        return 0.7

    def reason(self, context: Dict[str, Any]) -> MentalModelResult:
        goal = context.get("goal", context.get("input", ""))
        reasoning = (
            f"Applying first-principles thinking to: {goal}\n"
            f"1. Identify the core problem without assumptions\n"
            f"2. Break it down to fundamental truths\n"
            f"3. Build a solution from those truths upward"
        )
        return MentalModelResult(
            model_name=self.name,
            reasoning=reasoning,
            explanation="First-principles reasoning strips away assumptions and rebuilds understanding from basic truths.",
        )


class RootCauseAnalysis(MentalModel):
    def __init__(self) -> None:
        super().__init__(
            name="root_cause_analysis",
            description="Identify the underlying cause of a problem rather than treating symptoms.",
        )

    def applicability(self, context: Dict[str, Any]) -> float:
        return 0.6

    def reason(self, context: Dict[str, Any]) -> MentalModelResult:
        goal = context.get("goal", context.get("input", ""))
        reasoning = (
            f"Performing root cause analysis on: {goal}\n"
            f"1. Identify the symptoms\n"
            f"2. Trace each symptom to its direct cause\n"
            f"3. Ask 'why' iteratively to reach the root cause\n"
            f"4. Verify the root cause with evidence"
        )
        return MentalModelResult(
            model_name=self.name,
            reasoning=reasoning,
            explanation="Root cause analysis digs past symptoms to find the fundamental cause.",
        )


class SecondOrderThinking(MentalModel):
    def __init__(self) -> None:
        super().__init__(
            name="second_order_thinking",
            description="Consider the downstream consequences of decisions beyond the immediate effects.",
        )

    def applicability(self, context: Dict[str, Any]) -> float:
        return 0.65

    def reason(self, context: Dict[str, Any]) -> MentalModelResult:
        goal = context.get("goal", context.get("input", ""))
        reasoning = (
            f"Applying second-order thinking to: {goal}\n"
            f"First-order effects: What happens immediately?\n"
            f"Second-order effects: What happens as a consequence?\n"
            f"Third-order effects: What are the long-term ripple effects?"
        )
        return MentalModelResult(
            model_name=self.name,
            reasoning=reasoning,
            explanation="Second-order thinking looks beyond immediate results to downstream consequences.",
        )


class SWOT(MentalModel):
    def __init__(self) -> None:
        super().__init__(
            name="swot",
            description="Evaluate Strengths, Weaknesses, Opportunities, and Threats.",
        )

    def applicability(self, context: Dict[str, Any]) -> float:
        return 0.5

    def reason(self, context: Dict[str, Any]) -> MentalModelResult:
        goal = context.get("goal", context.get("input", ""))
        reasoning = (
            f"SWOT analysis for: {goal}\n"
            f"Strengths: Internal capabilities that help achieve the goal\n"
            f"Weaknesses: Internal limitations that hinder progress\n"
            f"Opportunities: External factors that could be leveraged\n"
            f"Threats: External risks that could derail the effort"
        )
        return MentalModelResult(
            model_name=self.name,
            reasoning=reasoning,
            explanation="SWOT analysis provides a structured view of internal and external factors.",
        )


class OODALoop(MentalModel):
    def __init__(self) -> None:
        super().__init__(
            name="ooda_loop",
            description="Observe, Orient, Decide, Act — a循环 for rapid decision-making.",
        )

    def applicability(self, context: Dict[str, Any]) -> float:
        return 0.6

    def reason(self, context: Dict[str, Any]) -> MentalModelResult:
        goal = context.get("goal", context.get("input", ""))
        reasoning = (
            f"Applying OODA loop to: {goal}\n"
            f"Observe: Gather information about the current situation\n"
            f"Orient: Analyze the information in context\n"
            f"Decide: Choose a course of action\n"
            f"Act: Execute the decision and observe the results"
        )
        return MentalModelResult(
            model_name=self.name,
            reasoning=reasoning,
            explanation="The OODA loop emphasizes rapid iteration of observation and action.",
        )


class BayesianUpdating(MentalModel):
    def __init__(self) -> None:
        super().__init__(
            name="bayesian_updating",
            description="Update beliefs proportionally to the strength of new evidence.",
        )

    def applicability(self, context: Dict[str, Any]) -> float:
        return 0.55

    def reason(self, context: Dict[str, Any]) -> MentalModelResult:
        goal = context.get("goal", context.get("input", ""))
        reasoning = (
            f"Applying Bayesian reasoning to: {goal}\n"
            f"Prior belief: What do we currently think?\n"
            f"New evidence: What new information is available?\n"
            f"Posterior: How should the belief change given the evidence?"
        )
        return MentalModelResult(
            model_name=self.name,
            reasoning=reasoning,
            explanation="Bayesian updating provides a mathematical framework for belief revision.",
        )


class MentalModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, MentalModel] = {}

    def register(self, model: MentalModel) -> None:
        self._models[model.name] = model

    def get(self, name: str) -> Optional[MentalModel]:
        return self._models.get(name)

    def candidates(self, context: Dict[str, Any], threshold: float = 0.0) -> List[MentalModel]:
        scored = [(m, m.applicability(context)) for m in self._models.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored if s >= threshold]

    def all(self) -> List[MentalModel]:
        return list(self._models.values())


def create_default_registry() -> MentalModelRegistry:
    registry = MentalModelRegistry()
    registry.register(FirstPrinciples())
    registry.register(RootCauseAnalysis())
    registry.register(SecondOrderThinking())
    registry.register(SWOT())
    registry.register(OODALoop())
    registry.register(BayesianUpdating())
    return registry
