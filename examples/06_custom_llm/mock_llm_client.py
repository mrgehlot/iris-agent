#!/usr/bin/env python3
"""
Custom LLM client example (mock echo client).
This runs without external API calls.
"""

import asyncio
from typing import Any, AsyncGenerator

from iris_agent import Agent, PromptRegistry


class LocalEchoClient:
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
        last = messages[-1]["content"] if messages else ""
        return {
            "content": f"Echo: {last}",
            "tool_calls": [],
            "message": None,
            "finish_reason": "stop",
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
    ) -> AsyncGenerator[Any, None]:
        await asyncio.sleep(0)
        if False:
            yield None


def main() -> int:
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a mock assistant.")

    client = LocalEchoClient()
    agent = Agent(llm_client=client, prompt_registry=prompts)

    response = agent.run("Hello from the mock client.")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
