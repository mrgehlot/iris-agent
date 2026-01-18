#!/usr/bin/env python3
"""
Sync streaming agent example.
Set OPENAI_API_KEY and optionally OPENAI_MODEL/OPENAI_BASE_URL.
"""

import os

from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(config)
    agent = Agent(llm_client=client)

    for chunk in agent.run_stream("Tell me a short story about a robot."):
        print(chunk, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
