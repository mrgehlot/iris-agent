from __future__ import annotations

from dataclasses import dataclass
import os
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
    extra_body: Optional[dict] = None

    def __post_init__(self) -> None:
        """
        Provider-specific defaults for OpenAI-compatible Gemini endpoints.
        """
        if self.provider == LLMProvider.GOOGLE:
            if not self.base_url:
                self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")


class BaseLLMClient:
    """Shared helpers for LLM clients."""
    def __init__(self, config: LLMConfig) -> None:
        """
        Create a client from configuration.

        Args:
            config: LLM configuration including provider, model, and credentials.
        """
        self.config = config

    def _supports_json_response(self) -> bool:
        """
        Return True if response_format=json_object is supported.

        Returns:
            True when JSON response format is supported by the base URL.
        """
        if not self.config.base_url:
            return True
        return "googleapis" not in self.config.base_url

    def _build_completion_object(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        json_response: bool,
        seed: int | None,
        reasoning_effort: str | None,
        web_search_options: dict | None,
        extra_body: dict | None,
        *,
        stream: bool = False,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        completion_object: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if stream:
            completion_object["stream"] = True
        if tools:
            completion_object["tools"] = tools
            completion_object["tool_choice"] = "auto"
        if json_response and self._supports_json_response():
            completion_object["response_format"] = {"type": "json_object"}
        if seed is not None:
            completion_object["seed"] = seed
        if max_completion_tokens is not None:
            completion_object["max_completion_tokens"] = max_completion_tokens
        if max_tokens is not None:
            completion_object["max_tokens"] = max_tokens
        if reasoning_effort or self.config.reasoning_effort:
            completion_object["reasoning_effort"] = reasoning_effort or self.config.reasoning_effort
        if web_search_options or self.config.web_search_options:
            completion_object["web_search_options"] = web_search_options or self.config.web_search_options
        if extra_body or self.config.extra_body:
            completion_object["extra_body"] = extra_body or self.config.extra_body
        return completion_object

    @staticmethod
    def _normalize_response(response: Any) -> dict:
        message = response.choices[0].message
        return {
            "content": message.content,
            "tool_calls": message.tool_calls or [],
            "message": message,
            "finish_reason": response.choices[0].finish_reason,
        }


class AsyncLLMClient(BaseLLMClient):
    """Async OpenAI-backed LLM client with tool support."""
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

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
        extra_body: dict | None = None,
    ) -> dict:
        completion_object = self._build_completion_object(
            messages,
            tools,
            temperature,
            json_response,
            seed,
            reasoning_effort,
            web_search_options,
            extra_body,
            max_completion_tokens=max_completion_tokens,
        )
        response = await self._client.chat.completions.create(**completion_object)
        return self._normalize_response(response)

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
        extra_body: dict | None = None,
    ):
        completion_object = self._build_completion_object(
            messages,
            tools,
            temperature,
            json_response,
            seed,
            reasoning_effort,
            web_search_options,
            extra_body,
            stream=True,
            max_tokens=max_tokens,
        )
        response = await self._client.chat.completions.create(**completion_object)
        async for chunk in response:
            yield chunk


class SyncLLMClient(BaseLLMClient):
    """Sync OpenAI-backed LLM client with tool support."""
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from openai import OpenAI

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        json_response: bool = False,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        completion_object = self._build_completion_object(
            messages,
            tools,
            temperature,
            json_response,
            seed,
            reasoning_effort,
            web_search_options,
            extra_body,
            max_completion_tokens=max_completion_tokens,
        )
        response = self._client.chat.completions.create(**completion_object)
        return self._normalize_response(response)

    def chat_completion_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        json_response: bool = False,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
        extra_body: dict | None = None,
    ):
        completion_object = self._build_completion_object(
            messages,
            tools,
            temperature,
            json_response,
            seed,
            reasoning_effort,
            web_search_options,
            extra_body,
            stream=True,
            max_tokens=max_tokens,
        )
        response = self._client.chat.completions.create(**completion_object)
        for chunk in response:
            yield chunk
