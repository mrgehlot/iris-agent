# Quick Start Guide

Get started with Iris Agent in 5 minutes!

## Installation

```bash
pip install iris-agent
```

## Basic Usage

### 1. Simple Agent

```python
from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry

# Setup
prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are a helpful assistant.")

llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key="your-api-key"
)
llm_client = BaseLLMClient(llm_config)

# Create agent
agent = Agent(
    llm_client=llm_client,
    prompt_registry=prompts,
)

# Use it
response = agent.run("Hello!")
print(response)
```

### 2. Agent with Tools

```python
from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry, ToolRegistry, tool

# Setup prompts
prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are a helpful math assistant.")

# Setup tools
tools = ToolRegistry()

@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

tools.register(add)

# Create agent
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key="your-api-key"
)
llm_client = BaseLLMClient(llm_config)

agent = Agent(
    llm_client=llm_client,
    prompt_registry=prompts,
    tool_registry=tools,
)

# Use it
response = agent.run("What is 5 + 3? Use the add tool.")
print(response)
```

### 3. Async Agent

```python
import asyncio
from iris_agent import AsyncAgent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry

async def main():
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)
    
    agent = AsyncAgent(
        llm_client=llm_client,
        prompt_registry=prompts,
    )
    
    # Run
    response = await agent.run("Hello!")
    print(response)
    
    # Stream
    async for chunk in agent.run_stream("Tell me a story"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### 4. With Rich Logging

```python
from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry

prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are helpful.")

llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    api_key="your-api-key"
)
llm_client = BaseLLMClient(llm_config)

agent = Agent(
    llm_client=llm_client,
    prompt_registry=prompts,
    enable_logging=True,  # Enable Rich logging
)

response = agent.run("Hello!")
```

### 5. Custom Messages

```python
from iris_agent import create_message, Role

# Text message
msg = create_message(Role.USER, "Hello")

# Message with images (for vision models)
vision_msg = create_message(
    Role.USER,
    "Describe this image",
    images=["https://example.com/image.jpg"]
)

# Message with name
named_msg = create_message(
    Role.USER,
    "Hello",
    name="John Doe"
)
```

## Examples

See `examples/system_prompt_example.py` for more detailed examples.

## Next Steps

- Read the [README.md](README.md) for full documentation
- Check [SETUP.md](SETUP.md) for development setup
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
