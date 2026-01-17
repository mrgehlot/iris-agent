from typing import Callable, Dict, Optional


class PromptRegistry:
    """Store and render named prompts."""
    def __init__(self) -> None:
        """Initialize empty prompt registry."""
        self._prompts: Dict[str, str | Callable[..., str]] = {}

    def add_prompt(self, name: str, template: str | Callable[..., str]) -> None:
        """
        Register a prompt as a string or callable.

        Args:
            name: Prompt identifier used when rendering.
            template: String template or callable returning a prompt string.
        """
        self._prompts[name] = template

    def get_prompt(self, name: str) -> Optional[str | Callable[..., str]]:
        """
        Fetch a prompt by name, or None if missing.

        Args:
            name: Prompt identifier.

        Returns:
            The stored prompt or None if not found.
        """
        return self._prompts.get(name)

    def render(self, prompt_name: str, **kwargs) -> Optional[str]:
        """
        Render a prompt with optional formatting kwargs.

        Args:
            prompt_name: Prompt identifier.
            **kwargs: Values passed to the prompt template or callable.

        Returns:
            Rendered prompt string or None if the prompt is missing.
        """
        prompt = self._prompts.get(prompt_name)
        if prompt is None:
            return None
        if callable(prompt):
            return prompt(**kwargs)
        return prompt.format(**kwargs) if kwargs else prompt
