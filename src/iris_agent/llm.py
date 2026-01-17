from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class LLMProvider:
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: Optional[str] = None
    web_search_options: Optional[dict] = None


class BaseLLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "Install openai optional dependency: pip install iris-agent-framework[openai]"
            ) from exc

        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    def _supports_json_response(self) -> bool:
        if not self.config.base_url:
            return True
        return "googleapis" not in self.config.base_url

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        json_response: bool = False,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
    ) -> dict:
        completion_object: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            completion_object["tools"] = tools
            completion_object["tool_choice"] = "auto"
        if json_response and self._supports_json_response():
            completion_object["response_format"] = {"type": "json_object"}
        if seed is not None:
            completion_object["seed"] = seed
        if max_completion_tokens is not None:
            completion_object["max_completion_tokens"] = max_completion_tokens
        if reasoning_effort or self.config.reasoning_effort:
            completion_object["reasoning_effort"] = reasoning_effort or self.config.reasoning_effort
        if web_search_options or self.config.web_search_options:
            completion_object["web_search_options"] = web_search_options or self.config.web_search_options

        response = await self._client.chat.completions.create(**completion_object)
        message = response.choices[0].message
        return {
            "content": message.content,
            "tool_calls": message.tool_calls or [],
            "message": message,
            "finish_reason": response.choices[0].finish_reason,
        }

    async def chat_completion_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        json_response: bool = False,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
    ):
        completion_object: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            completion_object["tools"] = tools
            completion_object["tool_choice"] = "auto"
        if json_response and self._supports_json_response():
            completion_object["response_format"] = {"type": "json_object"}
        if seed is not None:
            completion_object["seed"] = seed
        if max_tokens is not None:
            completion_object["max_tokens"] = max_tokens
        if reasoning_effort or self.config.reasoning_effort:
            completion_object["reasoning_effort"] = reasoning_effort or self.config.reasoning_effort
        if web_search_options or self.config.web_search_options:
            completion_object["web_search_options"] = web_search_options or self.config.web_search_options

        response = await self._client.chat.completions.create(**completion_object)
        async for chunk in response:
            yield chunk
