from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class LLMProvider:
    """Supported provider identifiers."""
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: Optional[str] = None
    web_search_options: Optional[dict] = None


class BaseLLMClient:
    """OpenAI-backed LLM client with tool support."""
    def __init__(self, config: LLMConfig) -> None:
        """
        Create a client from configuration.

        Args:
            config: LLM configuration including provider, model, and credentials.
        """
        from openai import AsyncOpenAI

        self.config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    def _supports_json_response(self) -> bool:
        """
        Return True if response_format=json_object is supported.

        Returns:
            True when JSON response format is supported by the base URL.
        """
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
        """
        Call chat completions and return a normalized response.

        Args:
            messages: Conversation history list of message dicts.
            tools: Optional OpenAI-compatible tool schemas.
            temperature: Sampling temperature.
            json_response: Request JSON-only response when supported.
            max_completion_tokens: Cap the completion token count.
            seed: Optional seed for deterministic sampling.
            reasoning_effort: Optional reasoning effort hint for supported models.
            web_search_options: Optional web search options for supported models.

        Returns:
            A dict with content, tool_calls, message, and finish_reason.
        """
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
        """
        Stream chat completions from the provider.

        Args:
            messages: Conversation history list of message dicts.
            tools: Optional OpenAI-compatible tool schemas.
            temperature: Sampling temperature.
            json_response: Request JSON-only response when supported.
            max_tokens: Cap the streamed completion tokens.
            seed: Optional seed for deterministic sampling.
            reasoning_effort: Optional reasoning effort hint for supported models.
            web_search_options: Optional web search options for supported models.

        Yields:
            Streaming chunks from the provider client.
        """
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
