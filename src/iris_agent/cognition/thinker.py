from __future__ import annotations

from typing import Any, Dict, List

from .base import CognitiveModule
from .events import CognitiveEventType
from .mental_models import MentalModel, MentalModelRegistry, create_default_registry


class Thinker(CognitiveModule):
    def __init__(
        self,
        registry: MentalModelRegistry | None = None,
        threshold: float = 0.3,
        max_models: int = 3,
    ) -> None:
        super().__init__()
        self._registry = registry or create_default_registry()
        self._threshold = threshold
        self._max_models = max_models

    @property
    def registry(self) -> MentalModelRegistry:
        return self._registry

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self._registry.candidates(context, self._threshold)
        selected = candidates[: self._max_models]

        if not selected:
            context["reasoning_context"] = ""
            context["selected_mental_model"] = "none"
            context["mental_model_explanation"] = ""
            return context

        results = []
        for model in selected:
            result = model.reason(context)
            results.append(result)

        combined_reasoning = "\n\n".join(
            f"[{r.model_name}]\n{r.reasoning}" for r in results
        )
        combined_explanation = " | ".join(
            f"{r.model_name}: {r.explanation}" for r in results
        )

        context["reasoning_context"] = combined_reasoning
        context["selected_mental_model"] = results[0].model_name
        context["mental_model_explanation"] = combined_explanation

        if self.mind:
            self.mind.event_bus.emit(
                CognitiveEventType.MENTAL_MODEL_SELECTED,
                models=[r.model_name for r in results],
                explanations=[r.explanation for r in results],
            )

        return context
