# How-To Guides

This section provides practical guides for common tasks.

## How to Define Tools

Tools are the primary way your agent interacts with the world.

### 1. Simple Function Tool

```python
import os
from iris_agent import tool, ToolRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Register the tool
tool_registry = ToolRegistry()
tool_registry.register(add)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful math assistant.")

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry,
    tool_registry=tool_registry
)

# Use the agent with tools
response = agent.run("What is 12 + 30? Use the add tool.")
print(response)
```

### 2. Tool with Complex Types

Iris Agent supports `List`, `Dict`, and `Literal` for validation.

```python
import os
from typing import Literal, List
from iris_agent import tool, ToolRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

@tool
def search_products(
    query: str,
    category: Literal["electronics", "clothing", "books"],
    tags: List[str] = None
) -> str:
    """Search for products in a specific category."""
    # Implementation...
    return f"Found 5 items in {category} matching '{query}'"

# Register the tool
tool_registry = ToolRegistry()
tool_registry.register(search_products)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful shopping assistant.")

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry,
    tool_registry=tool_registry
)

# Use the agent with tools
response = agent.run("Search for laptops in electronics category.")
print(response)
```

### 3. Registering Multiple Tools

```python
import os
from iris_agent import ToolRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Register all tools
tool_registry = ToolRegistry()
tool_registry.register(add)
tool_registry.register(multiply)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful math assistant.")

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry,
    tool_registry=tool_registry
)

# Use the agent with multiple tools
response = agent.run("Calculate 5 * 7 and then add 10 to the result.")
print(response)
```

### 4. Registering Tools from a Class

You can organize tools in a class and register all of them at once using `register_from()`:

```python
import os
from iris_agent import ToolRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry, tool

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
    
    @tool
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Create an instance of the class
math_tools = MathTools()

# Register all tools from the class at once
tool_registry = ToolRegistry()
tool_registry.register_from(math_tools)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful math assistant with access to various math operations.")

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry,
    tool_registry=tool_registry
)

# Use the agent with all the class tools
response = agent.run("Calculate (10 + 5) * 3 - 7, then divide the result by 2.")
print(response)
```

**Note:** `register_from()` automatically scans the object for all methods decorated with `@tool` and registers them. This is useful for organizing related tools together.

## How to Stream Responses

Streaming is essential for a responsive UI. You can stream with `AsyncAgent`
or use `Agent.run_stream()` for a sync-friendly iterator.

### Async Streaming

```python
import asyncio
import os
from iris_agent import AsyncAgent, AsyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

async def stream_chat():
    # Create client
    client = AsyncLLMClient(LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    ))
    
    # Create prompt registry
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt("assistant", "You are a helpful assistant.")
    
    # Create agent
    agent = AsyncAgent(
        llm_client=client,
        prompt_registry=prompt_registry
    )

    # run_stream yields chunks of text
    async for chunk in agent.run_stream("Tell me a short story about a robot."):
        print(chunk, end="", flush=True)
    print()  # New line after streaming

asyncio.run(stream_chat())
```

### Sync Streaming

```python
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Create client
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))

# Create prompt registry
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

# Create agent
agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

# Stream responses
for chunk in agent.run_stream("Tell me a short story about a robot."):
    print(chunk, end="", flush=True)
print()  # New line after streaming
```

## How to Use JSON Mode

If you need the agent to output strict JSON, use the `json_response` parameter.

### Async JSON Mode

```python
import asyncio
import os
from iris_agent import AsyncAgent, AsyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

async def get_json_data():
    # Create client
    client = AsyncLLMClient(LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    ))
    
    # Create prompt registry
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt("assistant", "You are a helpful assistant that outputs JSON.")
    
    # Create agent
    agent = AsyncAgent(
        llm_client=client,
        prompt_registry=prompt_registry
    )
    
    prompt = "Generate a list of 3 cities with lat/long."
    
    # Ensure you mention JSON in the prompt as well for best results
    response = await agent.run(
        prompt + " Output JSON.",
        json_response=True
    )
    print(response)
    # Output: {"cities": [...]}

asyncio.run(get_json_data())
```

### Sync JSON Mode

```python
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Create client
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))

# Create prompt registry
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant that outputs JSON.")

# Create agent
agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

# Use JSON mode
response = agent.run(
    "Generate a list of 3 cities with lat/long. Output JSON.",
    json_response=True
)
print(response)
```

## How to Handle Images (Multimodal)

Use `create_message` to send images to models like GPT-4o or Gemini 1.5 Pro.

```python
import os
from iris_agent import create_message, Role, Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Create client (GPT-4o or Gemini 1.5 Pro support images)
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",  # Use gpt-4o for image support
    api_key=os.getenv("OPENAI_API_KEY")
))

# Create prompt registry
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant that can analyze images.")

# Create agent
agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

# Create a message with an image
msg = create_message(
    role=Role.USER,
    content="What is in this image?",
    images=["https://example.com/photo.jpg"]
)

# Send the message to the agent
response = agent.run(msg)
print(response)
```

### Multiple Images

```python
# You can also send multiple images
msg = create_message(
    role=Role.USER,
    content="Compare these two images.",
    images=[
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg"
    ]
)

response = agent.run(msg)
print(response)
```

### Image-Only Messages

```python
# For image-only messages (no text), use empty content
msg = create_message(
    role=Role.USER,
    content="",  # Empty string for image-only
    images=["https://example.com/chart.png"]
)

response = agent.run(msg)
print(response)
```

## How to Use Gemini OpenAI Compatibility

Gemini models are accessible through the OpenAI-compatible endpoint. When you set
`provider=LLMProvider.GOOGLE`, Iris Agent defaults to the Gemini OpenAI base URL
and reads `GEMINI_API_KEY` if `api_key` is not provided.

### Basic Gemini Usage

```python
import os
from iris_agent import Agent, LLMConfig, LLMProvider, SyncLLMClient, PromptRegistry

# Create Gemini client
config = LLMConfig(
    provider=LLMProvider.GOOGLE,
    model="gemini-3-flash-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
)
client = SyncLLMClient(config)

# Create prompt registry
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

# Create agent
agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

response = agent.run("Explain how AI works.")
print(response)
```

### Gemini with Thinking Config

To send Gemini-specific fields (like `thinking_config`), pass `extra_body`:

```python
import os
from iris_agent import Agent, LLMConfig, LLMProvider, SyncLLMClient, PromptRegistry

# Create Gemini client
config = LLMConfig(
    provider=LLMProvider.GOOGLE,
    model="gemini-2.0-flash-exp",
    api_key=os.getenv("GEMINI_API_KEY"),
)
client = SyncLLMClient(config)

# Create prompt registry
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

# Create agent
agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

# Use with thinking config
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
print(response)
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
import os
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

registry = PromptRegistry()
registry.add_prompt("assistant", "You are a helpful AI assistant.")

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="assistant"
)

# Use the agent
response = agent.run("Hello, how are you?")
print(response)
```

### 2. Template Prompts (String Formatting)

You can use Python's `.format()` syntax to create dynamic prompts. Since agents render prompts without arguments, you need to pre-render template prompts:

**Option A: Pre-render and add to registry**

```python
import os
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

registry = PromptRegistry()
registry.add_prompt("personal_assistant", "You are {name}'s personal assistant. Today is {date}.")

# Render with variables
prompt_text = registry.render("personal_assistant", name="Alice", date="2024-01-15")
# Result: "You are Alice's personal assistant. Today is 2024-01-15."

# Add the rendered prompt back to the registry
registry.add_prompt("alice_assistant", prompt_text)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="alice_assistant"
)

# Use the agent
response = agent.run("What can you help me with today?")
print(response)
```

**Option B: Use callable prompts (recommended)**

```python
import os
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

def personal_assistant_prompt(name: str = "User", date: str = "") -> str:
    """Generate personalized assistant prompt."""
    date_str = f" Today is {date}." if date else ""
    return f"You are {name}'s personal assistant.{date_str}"

registry = PromptRegistry()
registry.add_prompt("personal_assistant", personal_assistant_prompt)

# Create client
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))

# Option 1: Use with defaults (if function has defaults)
agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="personal_assistant"
)

# Option 2: Render with specific values and add to registry
rendered = registry.render("personal_assistant", name="Alice", date="2024-01-15")
registry.add_prompt("alice_assistant", rendered)
agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="alice_assistant"
)

# Use the agent
response = agent.run("What can you help me with?")
print(response)
```

### 3. Callable Prompts (Function-Based)

For maximum flexibility, you can register a function that generates the prompt dynamically:

```python
import os
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

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

# Add the rendered prompt to registry
registry.add_prompt("bob_assistant", prompt)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="bob_assistant"
)

# Use the agent
response = agent.run("What can you help me with?")
print(response)
```

### 4. Using Multiple Prompts with Different Agents

The `PromptRegistry` integrates seamlessly with agents, allowing you to create multiple agents with different personas:

```python
import os
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

registry = PromptRegistry()
registry.add_prompt("coder", "You are an expert Python developer.")
registry.add_prompt("writer", "You are a creative writing assistant.")

# Create client
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))

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

# Use the agents
code_response = dev_agent.run("Write a Python function to calculate factorial.")
print("Coder:", code_response)

story_response = writer_agent.run("Write a short story about a robot.")
print("Writer:", story_response)
```

The agent automatically:
1. Looks up the prompt by `system_prompt_name`
2. Renders it (if it's a template or callable, you may need to provide kwargs)
3. Inserts it as the first message in memory (role: "developer")

### 5. Advanced Use Cases

#### Multi-Tenant Applications

```python
import os
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

def tenant_prompt(tenant_id: str, tenant_config: dict) -> str:
    """Generate prompt based on tenant configuration."""
    style = tenant_config.get("style", "professional")
    domain = tenant_config.get("domain", "general")
    return f"You are a {style} assistant specializing in {domain} for tenant {tenant_id}."

registry = PromptRegistry()
registry.add_prompt("tenant", tenant_prompt)

# Create client
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))

# Use in a web application
def get_tenant_config(tenant_id: str):
    # Mock function - replace with actual config retrieval
    return {"style": "professional", "domain": "e-commerce"}

def get_agent_for_tenant(tenant_id: str):
    config = get_tenant_config(tenant_id)
    # Render prompt with tenant-specific config
    rendered = registry.render("tenant", tenant_id=tenant_id, tenant_config=config)
    registry.add_prompt(f"tenant_{tenant_id}", rendered)
    
    return Agent(
        llm_client=client,
        prompt_registry=registry,
        system_prompt_name=f"tenant_{tenant_id}"
    )

# Use it
agent = get_agent_for_tenant("tenant_123")
response = agent.run("How can you help me?")
print(response)
```

#### A/B Testing Prompts

```python
import os
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

registry = PromptRegistry()
registry.add_prompt("variant_a", "You are a concise assistant.")
registry.add_prompt("variant_b", "You are a detailed, thorough assistant.")

# Create client
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))

# Switch between variants based on user
user_id = 123  # Example user ID
variant = "variant_a" if user_id % 2 == 0 else "variant_b"
agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name=variant
)

# Use the agent
response = agent.run("Explain quantum computing.")
print(response)
```

#### Context-Aware Prompts

```python
import os
from datetime import datetime
from iris_agent import PromptRegistry, Agent, SyncLLMClient, LLMConfig, LLMProvider

def contextual_prompt(current_time: str, user_location: str) -> str:
    return f"""You are a helpful assistant.
Current time: {current_time}
User location: {user_location}
Adjust your responses based on timezone and location."""

registry = PromptRegistry()
registry.add_prompt("contextual", contextual_prompt)

# Render with current context
prompt = registry.render(
    "contextual",
    current_time=datetime.now().isoformat(),
    user_location="New York"
)

# Add rendered prompt to registry
registry.add_prompt("contextual_ny", prompt)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
agent = Agent(
    llm_client=client,
    prompt_registry=registry,
    system_prompt_name="contextual_ny"
)

# Use the agent
response = agent.run("What's the weather like?")
print(response)
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

## How to Manage Memory

Iris Agent automatically manages conversation history in the `memory` attribute. You can inspect, seed, and clear memory as needed.

### Inspecting Memory

```python
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

# Check initial memory (contains system prompt)
print("Initial memory:", agent.memory)

# Run a conversation
response = agent.run("Hello!")
print("Response:", response)

# Check memory after conversation
print("Memory after conversation:", agent.memory)
print("Memory size:", len(agent.memory))
```

### Seeding Memory

You can add messages to memory before starting a conversation:

```python
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry, Role, create_message

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

# Seed memory with previous context
agent.memory.append(create_message(Role.USER, "My name is Alice and I love Python."))
agent.memory.append(create_message(Role.ASSISTANT, "Nice to meet you, Alice!"))

# Now continue the conversation
response = agent.run("What's my favorite programming language?")
print(response)  # The agent will remember Alice loves Python
```

### Clearing Memory

You can clear the conversation history while keeping the system prompt:

```python
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry
)

# Have a conversation
agent.run("Hello!")
agent.run("How are you?")

print("Memory before clear:", len(agent.memory))

# Clear conversation history (system prompt remains)
agent.memory.clear()

# Re-add system prompt if needed
agent._ensure_system_prompt()

print("Memory after clear:", len(agent.memory))
```

## How to Enable Logging

Iris Agent includes built-in Rich logging for beautiful, colorized terminal output showing the agent's step-by-step process.

### Basic Logging

```python
import os
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Create client
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))

# Create prompt registry
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

# Enable logging by setting enable_logging=True
agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry,
    enable_logging=True  # Enable Rich logging
)

# Run the agent - you'll see colorized output showing:
# - User messages (cyan)
# - LLM calls (dim)
# - Tool calls (magenta)
# - Tool responses (green)
# - Assistant responses (bold green)
response = agent.run("Summarize the benefits of good logging in one sentence.")
print(response)
```

### Custom Logger

You can also provide a custom logger instance:

```python
import os
import logging
from iris_agent import Agent, SyncLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Create a custom logger
custom_logger = logging.getLogger("my_agent")
custom_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
custom_logger.addHandler(handler)

# Create client and agent
client = SyncLLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
))
prompt_registry = PromptRegistry()
prompt_registry.add_prompt("assistant", "You are a helpful assistant.")

# Use custom logger
agent = Agent(
    llm_client=client,
    prompt_registry=prompt_registry,
    enable_logging=True,
    logger=custom_logger
)

response = agent.run("Hello!")
print(response)
```
