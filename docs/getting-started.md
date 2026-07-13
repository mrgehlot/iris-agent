# Getting Started

This guide will help you build your first cognitive agent using Iris Agent.

## Prerequisites

- Python 3.10 or higher.
- An `api_key` if you want to use OpenAI or Google Gemini.
- An `api_key` with `base_url` if you want to use other model providers or local models compatible with the OpenAI SDK.

## 1. Installation

```bash
pip install iris-agent
```

## 2. Create a Client

All agents share the same underlying LLM infrastructure.

```python
import os
from iris_agent import LLMConfig, LLMProvider, SyncLLMClient

api_key = os.getenv("OPENAI_API_KEY")

config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=api_key
)

client = SyncLLMClient(config)
```

## 3. Your First Cognitive Agent

`Mind` is the cognitive orchestrator. It runs a pipeline of specialized modules — Observer, Thinker, Planner, Critic, Reflector — and optionally maintains a structured World Model across turns.

```python
from iris_agent.cognition import Mind

mind = Mind(llm_client=client)

result = mind.run("What is the capital of France?")
print(result.response)
# The capital of France is Paris.
print(result.confidence)
# 0.85
print(result.mental_model)
# first_principles
```

`Mind` returns a `MindResult` with `.response`, `.confidence`, `.mental_model`, and the full `.context` dict.

## 4. Add Tools

Use the `@tool` decorator and pass a `ToolRegistry` to `Mind`. Core file-system tools are available via `include_core()`.

```python
from iris_agent import tool, ToolRegistry

tools = ToolRegistry()
tools.include_core()  # read_file, list_dir, glob_files, grep_files, run_command

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 25°C."

tools.register(get_weather)

mind = Mind(
    llm_client=client,
    tool_registry=tools
)

result = mind.run("What's the weather in Tokyo?")
print(result.response)
```

## 5. Persist Knowledge Across Turns

The World Model saves and loads structured knowledge (entities, relations, goals, lessons) to a JSON file, so each run builds on previous context.

```python
mind = Mind(
    llm_client=client,
    tool_registry=tools,
    world_model_path="./memory/world_model.json"
)

result1 = mind.run("List all .py files in the project.")
result2 = mind.run("Summarize what we learned about the codebase.")
# Second call carries forward entities extracted by the Observer
```

## 6. Async Support

For high-performance applications (web servers, concurrent requests), use `AsyncMind`:

```python
import asyncio
from iris_agent import AsyncLLMClient
from iris_agent.cognition import AsyncMind

async def main():
    client = AsyncLLMClient(config)
    mind = AsyncMind(
        llm_client=client,
        world_model_path="./memory/wm.json"
    )
    result = await mind.run("Tell me a quick joke.")
    print(result.response)

asyncio.run(main())
```

## 7. Streaming

Both `Mind` and `Agent` support streaming. Use `run_stream()` to receive tokens as they're generated:

```python
for chunk in mind.run_stream("Tell me a short story."):
    print(chunk, end="", flush=True)
print()
```

Async streaming works similarly:

```python
async for chunk in mind.run_stream("Tell me a short story."):
    print(chunk, end="", flush=True)
print()
```

---

## Alternative: Using the Simple Agent

If you don't need the cognitive pipeline, the classic `Agent` loop provides a lightweight alternative. It works the same way as `Mind` but skips the cognitive modules and World Model.

```python
from iris_agent import Agent, PromptRegistry

# Add a system prompt
prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are a friendly pirate assistant. Arr!")

agent = Agent(
    llm_client=client,
    prompt_registry=prompts,
    system_prompt_name="assistant"
)

response = agent.run("What is the capital of France?")
print(response)
# Arr! The capital of France be Paris, matey!
```

`Agent` supports all the same features — tools, streaming, async (`AsyncAgent`), JSON mode, and memory management. See the [How-To Guides](how-to.md) for details.

## Next Steps

- Dive into **[Core Concepts](concepts.md)** — the full cognitive pipeline and World Model.
- Explore **[Examples](examples.md)** including the basic Mind example.
- Check the **[API Reference](api.md)** for detailed documentation.
