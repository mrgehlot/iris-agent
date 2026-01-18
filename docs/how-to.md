# How-To Guides

This section provides practical guides for common tasks.

## How to Define Tools

Tools are the primary way your agent interacts with the world.

### 1. Simple Function Tool

```python
from iris_agent import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

### 2. Tool with Complex Types

Iris Agent supports `List`, `Dict`, and `Literal` for validation.

```python
from typing import Literal, List
from iris_agent import tool

@tool
def search_products(
    query: str,
    category: Literal["electronics", "clothing", "books"],
    tags: List[str] = None
) -> str:
    """Search for products in a specific category."""
    # Implementation...
    return "Found 5 items"
```

### 3. Registering Tools

```python
from iris_agent import ToolRegistry

registry = ToolRegistry()
registry.register(add)
registry.register(search_products)

# Pass registry to Agent
agent = Agent(..., tool_registry=registry)
```

## How to Stream Responses

Streaming is essential for a responsive UI. You can stream with `AsyncAgent`
or use `Agent.run_stream()` for a sync-friendly iterator.

```python
import asyncio
from iris_agent import AsyncAgent

async def stream_chat():
    agent = AsyncAgent(...)

    # run_stream yields chunks of text
    async for chunk in agent.run_stream("Tell me a story"):
        print(chunk, end="", flush=True)

asyncio.run(stream_chat())
```

Sync usage:

```python
from iris_agent import Agent

agent = Agent(...)

for chunk in agent.run_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

## How to Use JSON Mode

If you need the agent to output strict JSON, use the `json_response` parameter.

```python
async def get_json_data():
    agent = AsyncAgent(...)
    
    prompt = "Generate a list of 3 cities with lat/long."
    
    # Ensure you mention JSON in the prompt as well for best results
    response = await agent.run(
        prompt + " Output JSON.",
        json_response=True
    )
    print(response)
    # Output: {"cities": [...]}
```

## How to Handle Images (Multimodal)

Use `create_message` to send images to models like GPT-4o or Gemini 1.5 Pro.

```python
from iris_agent import create_message, Role

msg = create_message(
    role=Role.USER,
    content="What is in this image?",
    images=["https://example.com/photo.jpg"]
)

response = agent.run(msg)
```

## How to Use Gemini OpenAI Compatibility

Gemini models are accessible through the OpenAI-compatible endpoint. When you set
`provider=LLMProvider.GOOGLE`, Iris Agent defaults to the Gemini OpenAI base URL
and reads `GEMINI_API_KEY` if `api_key` is not provided.

```python
import os
from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider

config = LLMConfig(
    provider=LLMProvider.GOOGLE,
    model="gemini-3-flash-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
)
client = BaseLLMClient(config)
agent = Agent(llm_client=client)

response = agent.run("Explain how AI works.")
print(response)
```

To send Gemini-specific fields (like `thinking_config`), pass `extra_body`:

```python
response = agent.run(
    "Explain how AI works.",
    extra_body={
        "google": {
            "thinking_config": {
                "thinking_budget": "low",
                "include_thoughts": True,
            }
        }
    },
)
```

## How to Use Prompt Registry

The `PromptRegistry` is a powerful system for managing system instructions, personas, and dynamic prompt generation. It supports three types of prompts: static strings, template strings with placeholders, and callable functions.

### Why Use PromptRegistry?

Instead of hardcoding system prompts in your agent initialization, the `PromptRegistry` provides:

- **Centralized Management**: All prompts in one place, easy to update and version.
- **Dynamic Generation**: Create prompts on-the-fly based on context, user data, or configuration.
- **Reusability**: Share the same registry across multiple agents with different personas.
- **Template Support**: Use Python's `.format()` syntax for simple variable substitution.

### 1. Static Prompts

The simplest form is a static string prompt:

```python
from iris_agent import PromptRegistry, Agent

registry = PromptRegistry()
registry.add_prompt("assistant", "You are a helpful AI assistant.")

# Use it with an agent
agent = Agent(..., prompt_registry=registry, system_prompt_name="assistant")
```

### 2. Template Prompts (String Formatting)

You can use Python's `.format()` syntax to create dynamic prompts:

```python
registry = PromptRegistry()
registry.add_prompt("personal_assistant", "You are {name}'s personal assistant. Today is {date}.")

# Render with variables
prompt_text = registry.render("personal_assistant", name="Alice", date="2024-01-15")
# Result: "You are Alice's personal assistant. Today is 2024-01-15."

# Use with agent
agent = Agent(..., prompt_registry=registry, system_prompt_name="personal_assistant")
# Note: For templates with variables, you may need to pre-render or use callable prompts
```

### 3. Callable Prompts (Function-Based)

For maximum flexibility, you can register a function that generates the prompt dynamically:

```python
def create_assistant_prompt(user_name: str, context: str = "") -> str:
    """Generate a personalized assistant prompt."""
    base = f"You are {user_name}'s assistant."
    if context:
        base += f" Context: {context}"
    return base

registry = PromptRegistry()
registry.add_prompt("assistant", create_assistant_prompt)

# Render with arguments
prompt = registry.render("assistant", user_name="Bob", context="coding session")
# Result: "You are Bob's assistant. Context: coding session"
```

### 4. Using Multiple Prompts with Different Agents

The `PromptRegistry` integrates seamlessly with agents, allowing you to create multiple agents with different personas:

```python
registry = PromptRegistry()
registry.add_prompt("coder", "You are an expert Python developer.")
registry.add_prompt("writer", "You are a creative writing assistant.")

# Agent 1: Uses "coder" prompt
dev_agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="coder"
)

# Agent 2: Uses "writer" prompt
writer_agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="writer"
)
```

The agent automatically:
1. Looks up the prompt by `system_prompt_name`
2. Renders it (if it's a template or callable, you may need to provide kwargs)
3. Inserts it as the first message in memory (role: "developer")

### 5. Advanced Use Cases

#### Multi-Tenant Applications

```python
def tenant_prompt(tenant_id: str, tenant_config: dict) -> str:
    """Generate prompt based on tenant configuration."""
    style = tenant_config.get("style", "professional")
    domain = tenant_config.get("domain", "general")
    return f"You are a {style} assistant specializing in {domain} for tenant {tenant_id}."

registry = PromptRegistry()
registry.add_prompt("tenant", tenant_prompt)

# Use in a web application
def get_agent_for_tenant(tenant_id: str):
    config = get_tenant_config(tenant_id)
    return Agent(
        llm_client=client,
        prompt_registry=registry,
        system_prompt_name="tenant"
    )
```

#### A/B Testing Prompts

```python
registry = PromptRegistry()
registry.add_prompt("variant_a", "You are a concise assistant.")
registry.add_prompt("variant_b", "You are a detailed, thorough assistant.")

# Switch between variants based on user
variant = "variant_a" if user_id % 2 == 0 else "variant_b"
agent = Agent(..., prompt_registry=registry, system_prompt_name=variant)
```

#### Context-Aware Prompts

```python
def contextual_prompt(current_time: str, user_location: str) -> str:
    return f"""You are a helpful assistant.
Current time: {current_time}
User location: {user_location}
Adjust your responses based on timezone and location."""

registry = PromptRegistry()
registry.add_prompt("contextual", contextual_prompt)

# Render with current context
from datetime import datetime
prompt = registry.render(
    "contextual",
    current_time=datetime.now().isoformat(),
    user_location="New York"
)
```

### Best Practices

1. **Name Your Prompts Clearly**: Use descriptive names like `"customer_support"` instead of `"prompt1"`.
2. **Keep Prompts Focused**: Each prompt should define a single persona or role.
3. **Use Callables for Complex Logic**: If you need database lookups, API calls, or complex string manipulation, use callable prompts.
4. **Document Your Prompts**: Consider maintaining a separate file or docstring explaining what each prompt does.
5. **Version Your Prompts**: For production systems, consider adding version numbers to prompt names (e.g., `"assistant_v2"`).

### Prompt Rendering Flow

When an agent is initialized or when you manually render:

```python
# 1. Lookup
prompt_template = registry.get_prompt("assistant")

# 2. Render (if needed)
if callable(prompt_template):
    rendered = prompt_template(**kwargs)  # Call the function
elif isinstance(prompt_template, str) and kwargs:
    rendered = prompt_template.format(**kwargs)  # Format string
else:
    rendered = prompt_template  # Use as-is

# 3. Use in agent memory
agent.memory[0] = {"role": "developer", "content": rendered}
```
