#!/usr/bin/env python3
"""
Simple async agent example.
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
    prompts.add_prompt("assistant", "You are a helpful assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(llm_config)

    agent = AsyncAgent(llm_client=client, prompt_registry=prompts)
    response = await agent.run("Explain what an async function is in one sentence.")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
