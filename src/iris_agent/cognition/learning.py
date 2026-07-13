from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from .base import CognitiveModule
from .confidence import confidence_from_repetition


class Learning(CognitiveModule):
    def __init__(self) -> None:
        super().__init__()
        self._strategy_scores: Dict[str, List[float]] = defaultdict(list)
        self._mental_model_success: Dict[str, List[bool]] = defaultdict(list)
        self._tool_success: Dict[str, List[bool]] = defaultdict(list)

    def record_outcome(
        self,
        category: str,
        key: str,
        success: bool,
    ) -> None:
        store = getattr(self, f"_{category}_success", None)
        if store is not None:
            store[key].append(success)

    def success_rate(self, category: str, key: str) -> float:
        store = getattr(self, f"_{category}_success", None)
        if store is None:
            return 0.0
        outcomes = store.get(key, [])
        if not outcomes:
            return 0.0
        successes = sum(1 for o in outcomes if o)
        return confidence_from_repetition(successes, len(outcomes))

    def best_strategy(self, category: str) -> str:
        store = getattr(self, f"_{category}_success", None)
        if store is None:
            return ""
        best_key = ""
        best_rate = -1.0
        for key, outcomes in store.items():
            if not outcomes:
                continue
            rate = sum(1 for o in outcomes if o) / len(outcomes)
            if rate > best_rate:
                best_rate = rate
                best_key = key
        return best_key

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context.get("selected_mental_model", "")
        if model:
            success = context.get("plan_score", 0.5) > 0.5
            self.record_outcome("mental_model", model, success)

        return context
