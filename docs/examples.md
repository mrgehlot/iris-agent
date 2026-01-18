# Examples

This page contains complete, runnable examples demonstrating how to use Iris Agent. All examples are located in the `examples/` directory of the repository.

## Quick Navigation

- [Basic Usage](#basic-usage)
- [Prompts](#prompts)
- [Tools](#tools)
- [Memory](#memory)
- [Multi-Agent](#multi-agent)
- [Streaming](#streaming)
- [Advanced Features](#advanced-features)

## Basic Usage

### Simple Synchronous Agent

The most basic example - create an agent and have a conversation.

**File:** `examples/01_basic/simple_agent.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a concise assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = SyncLLMClient(llm_config)

    agent = Agent(llm_client=client, prompt_registry=prompts)
    response = agent.run("Say hello in one sentence.")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Async Agent

Use `AsyncAgent` for async/await workflows, ideal for web servers.

**File:** `examples/01_basic/async_agent.py`

```python
#!/usr/bin/env python3
import asyncio
import os
from iris_agent import AsyncAgent, AsyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

async def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = AsyncLLMClient(llm_config)

    agent = AsyncAgent(llm_client=client, prompt_registry=prompts)
    response = await agent.run("Explain what an async function is in one sentence.")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

## Streaming

### Async Streaming

Stream responses in real-time with `AsyncAgent.run_stream()`.

**File:** `examples/01_basic/streaming_agent.py`

```python
#!/usr/bin/env python3
import asyncio
import os
from iris_agent import AsyncAgent, AsyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

async def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a friendly assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = AsyncLLMClient(llm_config)

    agent = AsyncAgent(llm_client=client, prompt_registry=prompts)
    async for chunk in agent.run_stream("Write a 3-sentence story about a robot."):
        print(chunk, end="", flush=True)
    print()
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

### Sync Streaming

Use `Agent.run_stream()` for synchronous streaming.

**File:** `examples/01_basic/sync_streaming_agent.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider

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
    client = SyncLLMClient(config)
    agent = Agent(llm_client=client)

    for chunk in agent.run_stream("Tell me a short story about a robot."):
        print(chunk, end="", flush=True)
    print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## Prompts

### Static Prompts

Register and use simple string prompts.

**File:** `examples/02_prompts/prompt_registry_basic.py`

```python
#!/usr/bin/env python3
from iris_agent import PromptRegistry

def main() -> int:
    prompts = PromptRegistry()
    prompts.add_prompt("greeting", "Hello {name}!")

    print(prompts.render("greeting", name="Iris"))
    print(prompts.render("missing") or "No prompt found.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Dynamic Prompts

Use callable functions to generate prompts dynamically.

**File:** `examples/02_prompts/prompt_registry_dynamic.py`

```python
#!/usr/bin/env python3
from iris_agent import PromptRegistry

def main() -> int:
    prompts = PromptRegistry()

    def assistant_for(name: str) -> str:
        return f"You are {name}'s assistant. Be concise."

    prompts.add_prompt("assistant", assistant_for)
    print(prompts.render("assistant", name="Abhishek"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## Tools

### Basic Tool Registration

Register a simple function as a tool.

**File:** `examples/03_tools/tool_registry_basic.py`

```python
#!/usr/bin/env python3
from iris_agent import ToolRegistry, tool

def main() -> int:
    registry = ToolRegistry()

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    registry.register(add)
    print("Schemas:", registry.schemas())
    print("add(2, 3) =", registry.call("add", a=2, b=3))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Using Tools with Agent

Create an agent that can use tools to perform actions.

**File:** `examples/03_tools/tool_agent_usage.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry, ToolRegistry, tool

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
    client = SyncLLMClient(llm_config)

    agent = Agent(llm_client=client, prompt_registry=prompts, tool_registry=tools)
    response = agent.run("What is 12 + 30? Use the add tool.")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Async Tools

Define and use async tools for I/O operations.

**File:** `examples/03_tools/async_tools.py`

```python
#!/usr/bin/env python3
import asyncio
from iris_agent import ToolRegistry, tool

async def main() -> int:
    registry = ToolRegistry()

    @tool(description="Async add")
    async def add_async(a: int, b: int) -> int:
        return a + b

    registry.register(add_async)
    result = await registry.call_async("add_async", a=5, b=7)
    print("add_async(5, 7) =", result)
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

### Custom Tool Schema

Override automatic schema inference with a custom schema.

**File:** `examples/03_tools/tool_schema_custom.py`

```python
#!/usr/bin/env python3
from iris_agent import ToolRegistry, tool

def main() -> int:
    registry = ToolRegistry()

    @tool(
        name="search_web",
        description="Search the web for a query.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    def search_web(query: str) -> str:
        return f"Results for: {query}"

    registry.register(search_web)
    print("Schemas:", registry.schemas())
    print("search_web:", registry.call("search_web", query="iris agent"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## Memory

### Memory Basics

Inspect, seed, and clear agent memory.

**File:** `examples/04_memory/memory_basics.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry, Role, create_message

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
    client = SyncLLMClient(llm_config)

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
```

## Multi-Agent

### Two Agents Chatting

Two agents with different personas having a conversation.

**File:** `examples/05_multi_agent/two_agents_chat.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

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
    client = SyncLLMClient(llm_config)

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
```

### Planner/Executor Pattern

One agent creates a plan, another executes it.

**File:** `examples/05_multi_agent/planner_executor.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

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
    client = SyncLLMClient(llm_config)

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
```

## Advanced Features

### Custom LLM Client

Create a mock LLM client for testing without API calls.

**File:** `examples/06_custom_llm/mock_llm_client.py`

```python
#!/usr/bin/env python3
import asyncio
from typing import Any, AsyncGenerator
from iris_agent import Agent, PromptRegistry

class LocalEchoClient:
    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        json_response: bool = False,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
    ) -> dict:
        last = messages[-1]["content"] if messages else ""
        return {
            "content": f"Echo: {last}",
            "tool_calls": [],
            "message": None,
            "finish_reason": "stop",
        }

    async def chat_completion_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        json_response: bool = False,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
    ) -> AsyncGenerator[Any, None]:
        await asyncio.sleep(0)
        if False:
            yield None

def main() -> int:
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a mock assistant.")

    client = LocalEchoClient()
    agent = Agent(llm_client=client, prompt_registry=prompts)

    response = agent.run("Hello from the mock client.")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Rich Logging

Enable beautiful, colorized terminal output showing agent steps.

**File:** `examples/07_logging/rich_logging.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = SyncLLMClient(llm_config)

    agent = Agent(
        llm_client=client,
        prompt_registry=prompts,
        enable_logging=True,
    )

    response = agent.run("Summarize the benefits of good logging in one sentence.")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Message Creation

Create messages with text, images, and names.

**File:** `examples/08_messages/create_message_examples.py`

```python
#!/usr/bin/env python3
from iris_agent import Role, create_message

def main() -> int:
    text_msg = create_message(Role.USER, "Hello")
    print("Text message:", text_msg)

    image_msg = create_message(
        Role.USER,
        "Describe this image",
        images=["https://example.com/image.jpg"],
    )
    print("Image message:", image_msg)

    named_msg = create_message(Role.USER, "Hello", name="John Doe")
    print("Named message:", named_msg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Gemini Integration

Use Google Gemini models via OpenAI-compatible endpoint.

**File:** `examples/09_gemini/gemini_basic.py`

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    if not api_key:
        print("Set GEMINI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant.")

    llm_config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    client = SyncLLMClient(llm_config)

    agent = Agent(llm_client=client, prompt_registry=prompts)
    response = agent.run("Say hello in one sentence.")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## Additional Examples

### JSON Mode

Force the agent to output valid JSON only.

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant that outputs JSON.")

    client = SyncLLMClient(LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=api_key
    ))

    agent = Agent(llm_client=client, prompt_registry=prompts)
    
    response = agent.run(
        "Generate a list of 3 cities with their countries. Output JSON.",
        json_response=True
    )
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Multimodal (Images)

Send images to vision-capable models.

```python
#!/usr/bin/env python3
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry, Role, create_message

def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant that can analyze images.")

    client = SyncLLMClient(LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",  # Use vision-capable model
        api_key=api_key
    ))

    agent = Agent(llm_client=client, prompt_registry=prompts)
    
    # Create message with image
    msg = create_message(
        role=Role.USER,
        content="What is in this image?",
        images=["https://example.com/photo.jpg"]
    )
    
    response = agent.run(msg)
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Tools from a Class

Register all tools from a class at once.

```python
#!/usr/bin/env python3
import os
from iris_agent import ToolRegistry, tool, Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

class MathTools:
    """A class containing math-related tools."""
    
    @tool
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    @tool
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    
    @tool
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    # Create instance and register all tools
    math_tools = MathTools()
    tool_registry = ToolRegistry()
    tool_registry.register_from(math_tools)

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful math assistant.")

    client = SyncLLMClient(LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=api_key
    ))

    agent = Agent(
        llm_client=client,
        prompt_registry=prompts,
        tool_registry=tool_registry
    )

    response = agent.run("Calculate (10 + 5) * 3 - 7")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Error Handling in Tools

Handle errors gracefully in tool functions.

```python
#!/usr/bin/env python3
import os
from iris_agent import ToolRegistry, tool, Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Returns error message if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running this example.")
        return 1

    tool_registry = ToolRegistry()
    tool_registry.register(divide)

    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful math assistant. If a tool returns an error, explain it to the user.")

    client = SyncLLMClient(LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=api_key
    ))

    agent = Agent(
        llm_client=client,
        prompt_registry=prompts,
        tool_registry=tool_registry
    )

    # The agent will handle the error gracefully
    response = agent.run("What is 10 divided by 0?")
    print(response)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## Running Examples

All examples can be run directly from the repository root:

```bash
# Basic examples
python examples/01_basic/simple_agent.py
python examples/01_basic/async_agent.py
python examples/01_basic/streaming_agent.py

# Tools
python examples/03_tools/tool_agent_usage.py

# Multi-agent
python examples/05_multi_agent/planner_executor.py

# And so on...
```

## Requirements

Most examples require:
- `OPENAI_API_KEY` environment variable
- Optional: `OPENAI_MODEL` (defaults are set in examples)
- Optional: `OPENAI_BASE_URL`

Gemini examples require:
- `GEMINI_API_KEY` environment variable
- Optional: `GEMINI_MODEL`
- Optional: `GEMINI_BASE_URL`

## Next Steps

- Read the **[Getting Started Guide](getting-started.md)** for a step-by-step introduction
- Explore **[Core Concepts](concepts.md)** to understand the architecture
- Check **[How-To Guides](how-to.md)** for practical recipes
- Review the **[API Reference](api.md)** for detailed documentation
