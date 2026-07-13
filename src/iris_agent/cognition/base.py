from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .mind import Mind
    from .async_mind import AsyncMind

MindOrAsync = "Mind | AsyncMind"


class CognitiveModule(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._mind: Optional[MindOrAsync] = None

    @property
    def mind(self) -> Optional[MindOrAsync]:
        return self._mind

    def attach(self, mind: MindOrAsync) -> None:
        self._mind = mind

    def detach(self) -> None:
        self._mind = None

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: Dict[str, CognitiveModule] = {}

    def register(self, name: str, module: CognitiveModule) -> None:
        self._modules[name] = module

    def get(self, name: str) -> Optional[CognitiveModule]:
        return self._modules.get(name)

    def list(self) -> List[str]:
        return list(self._modules.keys())

    def all(self) -> List[CognitiveModule]:
        return list(self._modules.values())
