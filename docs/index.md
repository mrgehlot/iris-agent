# Iris Agent Documentation

Welcome to the documentation for **Iris Agent**!

Iris Agent is a lightweight, flexible, and provider-agnostic framework for building AI agents in Python. It simplifies the process of creating agents that can use tools, manage memory, and interact with various LLM providers.

## Key Features

- **Provider Agnostic**: Works with OpenAI, Google Gemini, or any OpenAI-compatible API (like LocalAI or vLLM).
- **Tool Support**: Easy-to-use `@tool` decorator that automatically infers JSON schemas from Python type hints.
- **Async & Sync**: First-class async and sync agents.
- **Streaming**: Built-in support for streaming responses for real-time applications.
- **Memory Management**: Automatic conversation history management with support for custom memory stores.
- **Prompt Management**: Centralized `PromptRegistry` for reusable and template-based system prompts.
- **Type Safety**: Built with strict type hints for better developer experience and tooling support.
- **Logging**: Integrated Rich logging for beautiful, readable debug output.

## Architecture Overview

Iris Agent is built around a few core components:

```mermaid
graph TD
    A[Agent/AsyncAgent] --> B[SyncLLMClient/AsyncLLMClient]
    A --> C[ToolRegistry]
    A --> D[PromptRegistry]
    A --> E[Memory List]
    B --> F[LLM Provider API]
    C --> G[Python Functions]
```

- **Agent**: The central controller that orchestrates the LLM, tools, and memory.
- **LLM Client**: Handles communication with the AI provider.
- **Tool Registry**: Manages available tools and executes them when requested by the model.
- **Prompt Registry**: Stores system instructions and templates.

## Quick Start

Here is a minimal example to get you running in seconds:

```python
from iris_agent import Agent, LLMConfig, LLMProvider, PromptRegistry, SyncLLMClient

# 1. Configure the LLM
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    api_key="sk-..."
)
client = SyncLLMClient(config)

# 2. Create a prompt registry and add a system prompt
prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are a helpful AI assistant.")

# 3. Create the Agent with the system prompt
agent = Agent(
    llm_client=client,
    prompt_registry=prompts,
    system_prompt_name="assistant"
)

# 4. Run
response = agent.run("Hello! how are you doing today?")
print(response)
# I'm doing great! How about you?
```

## Documentation Map

Explore the detailed documentation:

- **[Getting Started](getting-started.md)**: Your first steps with Iris Agent.
- **[Installation](installation.md)**: Setup guide for different environments.
- **[Core Concepts](concepts.md)**: Deep dive into Agents, Tools, and Memory.
- **[How-To Guides](how-to.md)**: Practical recipes for common tasks (Tools, Streaming, etc.).
- **[Modules Reference](modules.md)**: Project structure overview.
- **[API Reference](api.md)**: Detailed class and function documentation.
- **[Examples](examples.md)**: Code examples for various use cases.
- **[Troubleshooting](troubleshooting.md)**: Solutions to common problems.
- **[FAQ](faq.md)**: Frequently asked questions.

## License

This project is licensed under the MIT License.
