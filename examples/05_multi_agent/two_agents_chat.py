#!/usr/bin/env python3
"""
Two agents exchanging messages in a short loop.
Requires OPENAI_API_KEY.
"""

import os

from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts_a = PromptRegistry()
    prompts_a.add_prompt("assistant", "You are Agent A. Be concise.")

    prompts_b = PromptRegistry()
    prompts_b.add_prompt("assistant", "You are Agent B. Ask clarifying questions.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(llm_config)

    agent_a = Agent(llm_client=client, prompt_registry=prompts_a)
    agent_b = Agent(llm_client=client, prompt_registry=prompts_b)

    message = "Discuss the pros and cons of remote work."
    for _ in range(3):
        reply_a = agent_a.run(message)
        print("\nAgent A:", reply_a)
        reply_b = agent_b.run(reply_a)
        print("\nAgent B:", reply_b)
        message = reply_b

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
