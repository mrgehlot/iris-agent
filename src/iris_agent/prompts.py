from typing import Callable, Dict, Optional


class PromptRegistry:
    def __init__(self) -> None:
        self._prompts: Dict[str, str | Callable[..., str]] = {}

    def add_prompt(self, name: str, template: str | Callable[..., str]) -> None:
        self._prompts[name] = template

    def get_prompt(self, name: str) -> Optional[str | Callable[..., str]]:
        return self._prompts.get(name)

    def render(self, name: str, **kwargs) -> Optional[str]:
        prompt = self._prompts.get(name)
        if prompt is None:
            return None
        if callable(prompt):
            return prompt(**kwargs)
        return prompt.format(**kwargs) if kwargs else prompt
