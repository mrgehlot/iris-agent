# Getting Started

This guide will help you build your first AI agent using Iris Agent.

## Prerequisites

- Python 3.10 or higher.
- An `api_key` if you want to use OpenAI or Google Gemini.
- An `api_key` with `base_url` if you want to use other model providers or local models which are compatible with OpenAI SDK.

## 1. Installation

First, install the package:

```bash
pip install iris-agent
```

## 2. Create a Client

To create an agent, you need an `LLMConfig` and a `SyncLLMClient`.

```python
import os
from iris_agent import LLMConfig, LLMProvider, SyncLLMClient

# Ideally, load API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")

config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=api_key
)

client = SyncLLMClient(config)
```

## 3. Add a System Prompt

Define system instructions using the `PromptRegistry` to customize your agent's behavior.

```python
from iris_agent import PromptRegistry

# Define system instructions
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a friendly pirate assistant. Arr!")
```

## 4. Creating Your First Agent

Combine the client with an `Agent` instance and the prompt registry.

```python
from iris_agent import Agent

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry,
    system_prompt_name="assistant"
)

response = agent.run("What is the capital of France?")
print(response)
# Output: Arr! The capital of France be Paris, matey!
```

## 5. Adding Tools

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
    tool_registry=tool_registry,
    prompt_registry=prompt_registry,
    system_prompt_name="assistant"
)

response = agent.run("What's the weather in Tokyo?")
print(response)
# Arr! The weather in Tokyo be sunny and a toasty 25°C, matey!
```

## 6. Async Support

For high-performance applications (e.g., web servers), use `AsyncAgent`.

```python
import asyncio
from iris_agent import AsyncAgent, AsyncLLMClient

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key
    )
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt("assistant", "You are a helpful assistant.")
    client = AsyncLLMClient(config)
    agent = AsyncAgent(
        llm_client=client, 
        prompt_registry=prompt_registry, 
        system_prompt_name="assistant"
        )
    response = await agent.run("Tell me a quick joke.")
    print(response)
    # Hey, what do you call a function that only tells the truth? A literal... function.



asyncio.run(main())
```

## 7. Streaming

Stream responses to get tokens as they're generated. Both sync and async agents support streaming.

**Sync Streaming:**

```python
from iris_agent import Agent

agent = Agent(llm_client=client)
for chunk in agent.run_stream("Tell me a short story."):
    print(chunk, end="", flush=True)
print()
```

**Async Streaming:**

```python
import asyncio
from iris_agent import AsyncAgent

async def main():
    agent = AsyncAgent(llm_client=client)
    async for chunk in agent.run_stream("Tell me a short story."):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

## Next Steps

- Learn about **[Core Concepts](concepts.md)** like Memory and Tools.
- See more **[Examples](examples.md)** including multi-agent setups.
- Check the **[API Reference](api.md)** for detailed documentation.
