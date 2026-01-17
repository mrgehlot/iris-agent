#!/usr/bin/env python3
"""
Gemini usage via OpenAI-compatible endpoint.
Set GEMINI_API_KEY, GEMINI_BASE_URL, and GEMINI_MODEL.
"""

import os

from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key or not base_url:
        print("Set GEMINI_API_KEY and GEMINI_BASE_URL before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(llm_config)

    agent = Agent(llm_client=client, prompt_registry=prompts)
    response = agent.run("Say hello in one sentence.")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
