# Iris Agent Framework

A lightweight Python AI agent framework for building autonomous agents.

## Features

- 🤖 **Simple Agent Interface** - Easy-to-use sync and async agent classes
- 🛠️ **Tool Decorators** - Expose Python functions as LLM tools with automatic schema inference
- 🔌 **Provider Agnostic** - Support for OpenAI, Google Gemini, and more
- 📝 **Code-Defined Prompts** - Manage prompts in code, no database needed
- 🎨 **Rich Logging** - Beautiful step-by-step logging with Rich
- 🔄 **Streaming Support** - Built-in streaming for real-time responses
- ✅ **Type Safe** - Full type hints and validation

## Installation

### From PyPI
```bash
pip install iris-agent
```

### From source
```bash
git clone https://github.com/yourusername/iris-agent.git
cd iris-agent
pip install -e .
```

## Quick Start
```python
from iris_agent import (
    Agent,
    LLMConfig,
    LLMProvider,
    BaseLLMClient,
    PromptRegistry,
    ToolRegistry,
    tool,
)

prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are a helpful assistant.")

tools = ToolRegistry()

@tool(description="Add two numbers.")
def add(a: int, b: int) -> int:
    return a + b

tools.register(add)

client = BaseLLMClient(
    LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="sk-...",
    )
)

agent = Agent(
    llm_client=client,
    prompt_registry=prompts,
    tool_registry=tools,
)

response = agent.run("What is 2 + 3?")
print(response)
```

## Tool Decorators
Use `@tool` to expose any function as a tool. The framework will infer a JSON
schema from function annotations, or you can pass a schema explicitly.

```python
@tool(name="search_web", description="Search the web", parameters={...})
def search_web(query: str) -> str:
    ...
```

## Prompt Registry
Define prompts in code:
```python
prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are an expert travel planner.")
```

### System Prompts
System prompts control the agent's behavior and personality. You can add them as simple strings or dynamic callables:

```python
# Simple string prompt
prompts = PromptRegistry()
prompts.add_prompt("assistant", "You are a helpful AI assistant.")

# Dynamic prompt with parameters
prompts.add_prompt(
    "customer_support",
    lambda user_name: f"You are a customer support agent for {user_name}."
)

# Multiple prompts for different agent types
prompts.add_prompt("coder", "You are an expert Python programmer.")
prompts.add_prompt("writer", "You are a creative writing assistant.")

# Create agent with specific prompt
agent = Agent(
    llm_client=client,
    prompt_registry=prompts,
    system_prompt_name="coder"  # Uses the "coder" prompt
)
```

See `examples/system_prompt_example.py` for more detailed examples.

## Providers
`LLMConfig` supports multiple providers:
- OpenAI
- Google Gemini
- Additional providers can be added by implementing `BaseLLMClient`.

## Logging (Rich)
You can enable step-by-step agent logging using the `rich` package:

```python
agent = Agent(
    llm_client=client,
    prompt_registry=prompts,
    tool_registry=tools,
    enable_logging=True,
)
```

Rich logging is included by default.

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run only unit tests (skip integration tests that require API keys)
pytest -m "not integration"

# Run only integration tests (requires OPENAI_API_KEY)
pytest -m integration

# Run with coverage
pytest --cov=iris_agent --cov-report=html
```

## Development

1. Clone the repository
2. Install in editable mode: `pip install -e ".[dev]"`
3. Make your changes
4. Run tests: `pytest`
5. Format code: `black .` and `isort .`

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
