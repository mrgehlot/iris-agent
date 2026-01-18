# Getting Started

This guide will help you build your first AI agent using Iris Agent.

## Prerequisites

- Python 3.10 or higher.
- An API key for an LLM provider (e.g., OpenAI, Google Gemini, DeepSeek, etc.).

## 1. Installation

First, install the package:

```bash
pip install iris-agent
```

## 2. Basic Configuration

To create an agent, you need an `LLMConfig` and a `BaseLLMClient`.

```python
import os
from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider

# Ideally, load API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")

config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=api_key
)

client = BaseLLMClient(config)
```

## 3. Creating Your First Agent

Combine the client with an `Agent` instance. You can also define a system prompt using the `PromptRegistry`.

```python
from iris_agent import Agent, PromptRegistry

# Optional: Define system instructions
registry = PromptRegistry()
registry.add_prompt("assistant", "You are a friendly pirate assistant. Arr!")

agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="assistant"
)

response = agent.run("What is the capital of France?")
print(response)
# Output: Arr! The capital of France be Paris, matey!
```

## 4. Adding Tools

Tools allow your agent to interact with the outside world. Use the `@tool` decorator.

```python
from iris_agent import tool, ToolRegistry

tool_registry = ToolRegistry()

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real app, call an API here
    return f"The weather in {location} is sunny and 25°C."

tool_registry.register(get_weather)

# Update the agent with the tool registry
agent = Agent(
    llm_client=client,
    tool_registry=tool_registry
)

response = agent.run("What's the weather in Tokyo?")
print(response)
# The weather in Tokyo is sunny and 25°C.
```

## 5. Async Support

For high-performance applications (e.g., web servers), use `AsyncAgent`.

```python
import asyncio
from iris_agent import AsyncAgent

async def main():
    agent = AsyncAgent(llm_client=client)
    response = await agent.run("Tell me a quick joke.")
    print(response)
    # Hey, what do you call a function that only tells the truth? A literal... function. Whatever.



asyncio.run(main())
```

## 6. Sync Streaming

If you prefer a synchronous interface but want streaming tokens, use
`Agent.run_stream()`:

```python
from iris_agent import Agent

agent = Agent(llm_client=client)
for chunk in agent.run_stream("Tell me a short story."):
    print(chunk, end="", flush=True)
print()
```

## Next Steps

- Learn about **[Core Concepts](concepts.md)** like Memory and Tools.
- See more **[Examples](examples.md)** including multi-agent setups.
- Check the **[API Reference](api.md)** for detailed documentation.
