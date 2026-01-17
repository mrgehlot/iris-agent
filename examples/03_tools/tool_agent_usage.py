#!/usr/bin/env python3
"""
Using tools with an Agent. Requires OPENAI_API_KEY.
"""

import os

from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry, ToolRegistry, tool


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful math assistant.")

    tools = ToolRegistry()

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    tools.register(add)

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(llm_config)

    agent = Agent(llm_client=client, prompt_registry=prompts, tool_registry=tools)
    response = agent.run("What is 12 + 30? Use the add tool.")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
