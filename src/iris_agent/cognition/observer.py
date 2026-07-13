from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import CognitiveModule
from .events import CognitiveEventType
from .types import CognitiveEventType as CET
from .world_model_aware import WorldModelAware


class Observer(CognitiveModule, WorldModelAware):
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_input = context.get("input", "")
        goal = context.get("goal", raw_input)

        observation_id = uuid.uuid4().hex[:12]

        if self.mind:
            self.mind.event_bus.emit(
                CognitiveEventType.OBSERVATION_CREATED,
                observation_id=observation_id,
                content=raw_input,
                source="user",
                timestamp=time.time(),
            )

        if self.world_model:
            self.world_model.ingest_message("user", raw_input)
            self.world_model.set_goal(goal)
            self._extract_entities(raw_input)

        context["observation_id"] = observation_id
        context["goal"] = goal
        context["processed_input"] = raw_input
        return context

    def _extract_entities(self, text: str) -> None:
        if not self.world_model:
            return
        for path in self._find_file_paths(text):
            self.world_model.add_entity("file", {"path": path, "content_hint": ""})
        for url in self._find_urls(text):
            self.world_model.add_entity("url", {"url": url})
        for tool_name in self._find_tool_mentions(text):
            self.world_model.add_entity(
                "tool_mention", {"name": tool_name}
            )

    @staticmethod
    def _find_file_paths(text: str) -> List[str]:
        paths: List[str] = []
        patterns = [
            r'(?:^|\s)(/[^\s]+\.[a-zA-Z]+)',       # /abs/path/file.ext
            r'(?:^|\s)(\.\.?/[^\s]*)',               # ./relative or ../relative
            r'(?:^|\s)(\*{1,2}/[^\s]*)',             # glob patterns like **/*.py
            r'(?:^|\s)([^\s]+\.[a-zA-Z]+(?:/[^\s]*)?)',  # file.ext or file.ext/path
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                p = m.group(1).rstrip(".,;:!?)")
                if p and len(p) > 2:
                    paths.append(p)
        return paths

    @staticmethod
    def _find_urls(text: str) -> List[str]:
        return re.findall(r'https?://[^\s]+', text)

    @staticmethod
    def _find_tool_mentions(text: str) -> List[str]:
        tools = {"read_file", "list_dir", "glob_files", "grep_files",
                 "run_command", "web_fetch", "search", "calculate"}
        found: List[str] = set()
        for t in tools:
            if t in text.lower():
                found.add(t)
        return list(found)
