from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .world_model import WorldModel


class WorldModelAware:
    def __init__(self) -> None:
        self._world_model: Optional[WorldModel] = None

    @property
    def world_model(self) -> Optional[WorldModel]:
        return self._world_model

    def set_world_model(self, world_model: WorldModel) -> None:
        self._world_model = world_model
