from __future__ import annotations

from typing import Any, Dict

from .base import CognitiveModule
from .events import CognitiveEventType
from .types import Experience
from .world_model_aware import WorldModelAware


class Reflector(CognitiveModule, WorldModelAware):
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get("goal", "")
        action = context.get("last_action", "")
        actual_outcome = context.get("response", "")
        expected_outcome = context.get("expected_outcome", "")

        if not action and not actual_outcome:
            return context

        lessons = self._extract_lessons(goal, action, actual_outcome, expected_outcome)
        mistakes = self._extract_mistakes(actual_outcome, expected_outcome)
        strategies = self._extract_strategies(lessons)

        context["lessons"] = lessons
        context["mistakes"] = mistakes
        context["strategies"] = strategies
        context["reflection_completed"] = True

        if self.world_model:
            for lesson in lessons:
                self.world_model.record_lesson(lesson)

        exp_id = context.get("experience_id", "")
        if exp_id and self.mind:
            store = getattr(self.mind, "experience_store", None)
            if store:
                exp = store.get(exp_id)
                if exp:
                    exp.lessons = lessons
                    exp.mistakes = mistakes
                    exp.successful_strategies = strategies
                    exp.actual_outcome = actual_outcome

        if self.mind:
            self.mind.event_bus.emit(
                CognitiveEventType.REFLECTION_COMPLETED,
                goal=goal,
                lessons=lessons,
                mistakes=mistakes,
                strategies=strategies,
            )
            for lesson in lessons:
                self.mind.event_bus.emit(
                    CognitiveEventType.LESSON_LEARNED,
                    lesson=lesson,
                )

        return context

    def _extract_lessons(
        self, goal: str, action: str, actual: str, expected: str
    ) -> list:
        if actual and expected and actual != expected:
            return [f"Actual outcome differed from expected for: {goal}"]
        return ["Completed goal successfully"]

    def _extract_mistakes(self, actual: str, expected: str) -> list:
        if actual and expected and actual != expected:
            return ["Outcome did not match expectations"]
        return []

    def _extract_strategies(self, lessons: list) -> list:
        if lessons:
            return ["Continue with current approach"]
        return []
