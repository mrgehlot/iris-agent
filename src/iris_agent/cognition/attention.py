from __future__ import annotations

from typing import Any, Dict, List

from .base import CognitiveModule


class Attention(CognitiveModule):
    def __init__(self, max_context_items: int = 5) -> None:
        super().__init__()
        self._max_items = max_context_items

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get("goal", "")
        input_text = context.get("input", "")
        plan = context.get("plan", [])
        reasoning = context.get("reasoning_context", "")

        summary_parts = []
        if goal:
            summary_parts.append(f"Goal: {goal}")
        if input_text:
            summary_parts.append(f"Input: {input_text[:200]}")
        if reasoning:
            summary_parts.append(f"Reasoning: {reasoning[:300]}")

        context["attention_summary"] = "\n".join(summary_parts)

        context["focused_input"] = self._summarize(
            input_text, goal, self._max_items
        )

        return context

    def _summarize(self, input_text: str, goal: str, max_items: int) -> str:
        return input_text
