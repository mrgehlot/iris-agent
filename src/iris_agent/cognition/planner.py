from __future__ import annotations

from typing import Any, Dict, List

from .base import CognitiveModule
from .types import PlanStep
from .world_model_aware import WorldModelAware


class Planner(CognitiveModule, WorldModelAware):
    def __init__(self, max_steps: int = 5) -> None:
        super().__init__()
        self._max_steps = max_steps

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get("goal", context.get("input", ""))
        reasoning = context.get("reasoning_context", "")

        if reasoning:
            prompt = (
                f"Given this reasoning context:\n{reasoning}\n\n"
                f"Decompose the following goal into at most {self._max_steps} actionable steps:\n{goal}"
            )
        else:
            prompt = (
                f"Decompose the following goal into at most {self._max_steps} actionable steps:\n{goal}"
            )

        steps = self._build_plan(prompt)

        if self.world_model:
            for step in steps:
                tid = self.world_model.add_task(step.action)
                if self.world_model.current_goal_id:
                    self.world_model.link(
                        tid, self.world_model.current_goal_id, "subgoal_of"
                    )

        plan_summary = "\n".join(
            f"{i+1}. {step.action}" for i, step in enumerate(steps)
        )

        context["plan"] = [
            {"action": step.action, "expected_outcome": step.expected_outcome}
            for step in steps
        ]
        context["plan_summary"] = plan_summary

        return context

    def _build_plan(self, prompt: str) -> List[PlanStep]:
        return [
            PlanStep(
                action=f"Step {i+1}",
                expected_outcome=f"Outcome of step {i+1}",
            )
            for i in range(3)
        ]
