#!/usr/bin/env python3
"""
Streaming response example using AsyncAgent.
"""

import asyncio
import os

from iris_agent import AsyncAgent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry


async def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a friendly assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(llm_config)

    agent = AsyncAgent(llm_client=client, prompt_registry=prompts)
    async for chunk in agent.run_stream("Write a 3-sentence story about a robot."):
        print(chunk, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
