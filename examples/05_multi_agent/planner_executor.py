#!/usr/bin/env python3
"""
Planner/Executor pattern with two agents.
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

    prompts = PromptRegistry()
    prompts.add_prompt(
        "planner_assistant",
        "You are a planning agent. Produce a short numbered plan.",
    )
    
    prompts.add_prompt(
        "executor_assistant",
        "You are an execution agent. Follow the plan and answer succinctly.",
    )

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = BaseLLMClient(llm_config)

    planner = Agent(
        llm_client=client,
        prompt_registry=prompts,
        system_prompt_name="planner_assistant",
    )
    executor = Agent(
        llm_client=client,
        prompt_registry=prompts,
        system_prompt_name="executor_assistant",
    )

    task = "Design a 1-day itinerary for Mumbai."
    plan = planner.run(task)
    print("\nPlan:\n", plan)

    response = executor.run(f"Task: {task}\nPlan:\n{plan}")
    print("\nExecution:\n", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
