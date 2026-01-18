# Core Concepts

Understanding how Iris Agent works under the hood will help you build more robust applications.

## The Agent Loop

At its core, an `Agent` (or `AsyncAgent`) implements a **Loop** that continues until the AI produces a final answer.

1.  **Receive Input**: The agent accepts a user message (text or multimodal).
2.  **Update Memory**: The message is added to the agent's conversation history (`self.memory`).
3.  **LLM Call**: The agent sends the full history + available tool definitions to the LLM.
4.  **Decision**: The LLM decides to either:
    *   **Respond**: Generate a text response.
    *   **Call Tool**: Request execution of a specific tool with specific arguments.
5.  **Execution (if Tool)**:
    *   The agent executes the requested Python function.
    *   The result is captured and added to memory as a `Tool` message.
    *   The loop repeats (Go to Step 3).
6.  **Final Output**: When the LLM generates a text response (and no more tool calls), the text is returned to the user.

## Memory

Iris Agent uses a simple list-based memory structure compatible with OpenAI's chat format.

- **System Message**: The initial instruction (e.g., "You are a helpful assistant"). Always kept at index 0.
- **User Message**: Input from the human.
- **Assistant Message**: Output from the AI.
- **Tool Message**: Results from function calls.

```python
[
  {"role": "developer", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the date?"},
  {"role": "assistant", "tool_calls": [...]},
  {"role": "tool", "tool_call_id": "...", "content": "2023-10-27"},
  {"role": "assistant", "content": "It is October 27, 2023."}
]
```

## Tools and Type Inference

One of the most powerful features of Iris Agent is how it handles tools. Instead of manually writing JSON schemas, you write standard Python functions with type hints.

### The `@tool` Decorator

When you decorate a function with `@tool`, Iris Agent inspects the signature:

```python
@tool
def calculate_tax(amount: float, rate: float = 0.1) -> float:
    """Calculate tax for a given amount."""
    return amount * rate
```

This is automatically converted to:
```json
{
  "name": "calculate_tax",
  "description": "Calculate tax for a given amount.",
  "parameters": {
    "type": "object",
    "properties": {
      "amount": {"type": "number"},
      "rate": {"type": "number"}
    },
    "required": ["amount"]
  }
}
```

The `ToolRegistry` handles this conversion and the subsequent execution validation.

## Registries

To keep your code organized, Iris Agent uses Registries.

- **`ToolRegistry`**: A collection of available tools. You can share registries between multiple agents.
- **`PromptRegistry`**: A collection of system prompts or templates. This allows you to dynamically switch personas or update prompts without changing application code.

## LLM Client Architecture

The `BaseLLMClient` is the interface between your agent and the AI provider. It abstracts away provider-specific details and provides a unified API.

### LLMConfig

The `LLMConfig` dataclass contains all the information needed to connect to an LLM:

- **`provider`**: The provider identifier (e.g., `LLMProvider.OPENAI`, `LLMProvider.GOOGLE`)
- **`model`**: The specific model name (e.g., `"gpt-4o"`, `"gpt-4o-mini"`)
- **`api_key`**: Your API key (can be `None` if using environment variables)
- **`base_url`**: Optional override for custom endpoints (useful for local models, proxies, or Gemini via OpenAI-compatible API)
- **`reasoning_effort`**: Optional hint for models that support reasoning (e.g., `"high"`, `"medium"`, `"low"`)
- **`web_search_options`**: Optional configuration for models that support web search
- **`extra_body`**: Optional provider-specific request overrides (e.g., Gemini `thinking_config`)

### Provider Compatibility

The `BaseLLMClient` uses the OpenAI SDK under the hood, which means it works with:
- **OpenAI**: Direct API access
- **Google Gemini**: Via OpenAI-compatible base URL (`https://generativelanguage.googleapis.com/v1beta/openai/`)
- **Local Models**: Any OpenAI-compatible API (Ollama, vLLM, LocalAI, etc.)

The client automatically handles differences in API capabilities (e.g., JSON mode support detection).

## Messages and Roles

Iris Agent uses a message-based conversation format compatible with OpenAI's chat completions API.

### Message Structure

Messages are dictionaries with a `role` field and optional `content`, `name`, `images`, and `tool_calls` fields:

```python
{
  "role": "user",
  "content": "Hello!",
  "name": "Alice",  # Optional: for multi-user conversations
  "images": [...]   # Optional: for multimodal messages
}
```

### Roles

The framework supports five roles (defined in `Role` class):

- **`SYSTEM` / `DEVELOPER`**: System instructions. The agent uses `DEVELOPER` by default for prompts from `PromptRegistry`. Both are treated identically and always kept at index 0 in memory.
- **`USER`**: Human input messages.
- **`ASSISTANT`**: AI-generated responses. Can include `tool_calls` when the model wants to use tools.
- **`TOOL`**: Results from function calls. Must include `tool_call_id` to link back to the original request.

### Creating Messages

Use the `create_message()` helper function to create properly formatted messages:

```python
from iris_agent import create_message, Role

# Simple text message
user_msg = create_message(Role.USER, "What's the weather?")

# Multimodal message with images
image_msg = create_message(
    role=Role.USER,
    content="Describe this image",
    images=["https://example.com/photo.jpg"]
)

# Image-only message (no text)
image_only = create_message(
    role=Role.USER,
    content="",  # Empty string for image-only
    images=["https://example.com/chart.png"]
)
```

## Tool Execution and Error Handling

When the LLM requests a tool call, the agent:

1. **Parses Arguments**: Extracts JSON arguments from the tool call
2. **Validates**: Checks arguments against the tool's schema (type checking, required fields, enum values)
3. **Executes**: Calls the Python function (sync or async)
4. **Handles Errors**: If an exception occurs, the error message is captured and sent back to the LLM as the tool response, allowing the model to retry or adjust

```python
# If a tool raises an exception:
try:
    result = await tool_registry.call_async("get_weather", location="Tokyo")
except Exception as exc:
    # The agent automatically wraps this as:
    # {"role": "tool", "content": f"Tool error: {exc}"}
```

This error handling allows the LLM to understand what went wrong and potentially try a different approach.

## Streaming Architecture

When using `run_stream()`, the agent handles streaming differently from `run()`:

1. **Stream Collection**: Instead of waiting for the full response, chunks are yielded immediately to the caller
2. **Tool Call Buffering**: Tool calls arrive incrementally in the stream. The agent buffers these chunks until the stream completes
3. **Tool Execution**: After the stream ends, if tool calls were detected, they are executed (just like in `run()`)
4. **Loop Continuation**: The agent continues the loop, sending tool results back and streaming the next response

This allows for real-time user feedback while still supporting the full tool-calling workflow.

## Advanced LLM Parameters

The agent supports several advanced parameters that can be passed to `run()` or `run_stream()`:

- **`json_response`**: Forces the model to output valid JSON only (requires model support)
- **`seed`**: Sets a random seed for deterministic outputs (useful for testing)
- **`max_completion_tokens` / `max_tokens`**: Limits the response length
- **`reasoning_effort`**: Hints to models like o1 about how much reasoning to perform
- **`web_search_options`**: Enables web search for models that support it

These parameters are passed through to the underlying LLM client and may not be supported by all providers.

## Logging

Iris Agent includes built-in support for Rich logging, which provides beautiful, colorized terminal output showing:

- User messages (cyan)
- LLM calls (dim)
- Tool calls (magenta)
- Tool responses (green)
- Assistant responses (bold green)
- Finish reasons (dim)

Enable logging by setting `enable_logging=True` when creating an agent:

```python
agent = Agent(..., enable_logging=True)
```

You can also provide a custom logger instance for integration with your existing logging infrastructure.

## Sync vs Async

- **`AsyncAgent`**: The "real" agent. It uses `asyncio` for non-blocking I/O (HTTP requests to LLMs, database calls in tools). Ideal for web servers (FastAPI, Quart). Supports both `run()` and `run_stream()`.
- **`Agent`**: A synchronous wrapper around `AsyncAgent`. It manages the event loop for you. Ideal for scripts, CLI tools, or data science notebooks where `async/await` syntax might be cumbersome. **Note**: `Agent.run_stream()` is not supported; use `AsyncAgent` for streaming.
