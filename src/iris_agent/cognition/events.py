from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from .types import CognitiveEventType

EventHandler = Callable[..., None]

logger = logging.getLogger("iris_agent.cognition.events")


@dataclass
class CognitiveEvent:
    type: CognitiveEventType
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._handlers: Dict[CognitiveEventType, List[EventHandler]] = {}

    def subscribe(self, event_type: CognitiveEventType, handler: EventHandler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: CognitiveEventType, handler: EventHandler) -> None:
        if event_type in self._handlers:
            self._handlers[event_type] = [h for h in self._handlers[event_type] if h is not handler]

    def emit(self, event_type: CognitiveEventType, **data: Any) -> None:
        event = CognitiveEvent(type=event_type, data=data)
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(f"Event handler failed for {event_type}")

    def clear(self) -> None:
        self._handlers.clear()
