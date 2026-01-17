import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

from .async_agent import AsyncAgent
from .llm import BaseLLMClient, LLMConfig
from .prompts import PromptRegistry
from .tools import ToolRegistry


"""
Example: How to add system prompts for an AI agent

# Method 1: Simple string prompt
from iris_agent import Agent, BaseLLMClient, LLMConfig, PromptRegistry

# Create a prompt registry and add a system prompt
prompt_registry = PromptRegistry()
prompt_registry.add_prompt(
    "assistant",
    "You are a helpful AI assistant that provides clear and concise answers."
)

# Create LLM client
llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)
llm_client = BaseLLMClient(llm_config)

# Create agent with the prompt registry
agent = Agent(
    llm_client=llm_client,
    prompt_registry=prompt_registry,
    system_prompt_name="assistant"  # This matches the prompt name in registry
)

# Method 2: Dynamic prompt with parameters
prompt_registry = PromptRegistry()
prompt_registry.add_prompt(
    "customer_support",
    lambda user_name: f"You are a customer support agent for {user_name}. "
                      f"Always be polite and professional."
)

agent = Agent(
    llm_client=llm_client,
    prompt_registry=prompt_registry,
    system_prompt_name="customer_support"
)

# Method 3: Multiple prompts for different agents
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")
prompt_registry.add_prompt("coder", "You are an expert Python programmer.")
prompt_registry.add_prompt("writer", "You are a creative writing assistant.")

# Use different prompts for different agents
coding_agent = Agent(
    llm_client=llm_client,
    prompt_registry=prompt_registry,
    system_prompt_name="coder"
)

writing_agent = Agent(
    llm_client=llm_client,
    prompt_registry=prompt_registry,
    system_prompt_name="writer"
)

# Method 4: Using AsyncAgent directly (for async contexts)
from iris_agent import AsyncAgent

async def main():
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt(
        "assistant",
        "You are a helpful AI assistant."
    )
    
    async_agent = AsyncAgent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="assistant"
    )
    
    response = await async_agent.run("Hello!")
    print(response)
"""


class Agent:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_registry: Optional[PromptRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt_name: str = "assistant",
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._async_agent = AsyncAgent(
            llm_client=llm_client,
            prompt_registry=prompt_registry,
            tool_registry=tool_registry,
            system_prompt_name=system_prompt_name,
            enable_logging=enable_logging,
            logger=logger,
        )

    @property
    def memory(self):
        return self._async_agent.memory

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return loop.create_task(coro)

    def run(self, user_message: str, temperature: float = 1.0) -> str:
        result = self._run_async(self._async_agent.run(user_message, temperature=temperature))
        if asyncio.isfuture(result):
            raise RuntimeError("Agent.run called inside an event loop. Use AsyncAgent.")
        return result

    def run_stream(self, user_message: str, temperature: float = 1.0) -> AsyncGenerator[str, None]:
        raise RuntimeError("Use AsyncAgent.run_stream for streaming responses.")

    def call_tool(self, name: str, **kwargs) -> Any:
        result = self._run_async(self._async_agent.call_tool(name, **kwargs))
        if asyncio.isfuture(result):
            raise RuntimeError("Agent.call_tool called inside an event loop. Use AsyncAgent.")
        return result
