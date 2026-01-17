#!/usr/bin/env python3
"""
Memory usage: inspect, seed, and clear agent memory.
"""

import os

from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry, Role, create_message


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY") or "dummy"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(llm_config)

    agent = Agent(llm_client=client, prompt_registry=prompts)

    print("Initial memory:", agent.memory)
    agent.memory.append(create_message(Role.USER, "Seeded message"))
    print("After seeding:", agent.memory)

    if os.getenv("OPENAI_API_KEY"):
        response = agent.run("Reply to the seeded message.")
        print("Assistant response:", response)
        print("Final memory size:", len(agent.memory))
    else:
        print("Set OPENAI_API_KEY to run the agent and see memory growth.")

    agent.memory.clear()
    print("After clear:", agent.memory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
