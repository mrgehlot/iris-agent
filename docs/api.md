# API Reference

## Agent

### `iris_agent.Agent`

Synchronous wrapper around `AsyncAgent` for non-async usage.

```python
class Agent:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_registry: Optional[PromptRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt_name: str = "assistant",
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None
```

**Parameters:**

- `llm_client` (`BaseLLMClient`): The initialized LLM client instance.
- `prompt_registry` (`PromptRegistry`, optional): Registry for system prompts. Defaults to a new empty registry.
- `tool_registry` (`ToolRegistry`, optional): Registry for tools. Defaults to a new empty registry.
- `system_prompt_name` (`str`, default="assistant"): The key to look up in the prompt registry for the initial system message.
- `enable_logging` (`bool`, default=False): If True, enables rich logging to stdout.
- `logger` (`logging.Logger`, optional): Custom logger instance.

#### Methods

**`run(user_message: str | dict) -> str`**

Send a message to the agent and get a response.

- `user_message`: The input text or a message dict (created via `create_message`).
- **Returns**: The assistant's response text.

**`call_tool(name: str, **kwargs) -> Any`**

Manually call a tool registered with the agent.

- `name`: Name of the tool.
- `kwargs`: Arguments for the tool.

### `iris_agent.AsyncAgent`

The core asynchronous agent class.

```python
class AsyncAgent:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_registry: Optional[PromptRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt_name: str = "assistant",
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None
```

Same parameters as `Agent`.

#### Methods

**`async run(user_message: str | dict, ...) -> str`**

Run a single turn of conversation.

- `user_message`: Input text or dict.
- `json_response` (`bool`): If True, requests JSON output from the LLM.
- `max_completion_tokens` (`int`): Cap the response length.
- `seed` (`int`): Deterministic sampling seed.
- `reasoning_effort` (`str`): Reasoning effort for models that support it (e.g. "high", "medium", "low").
- `web_search_options` (`dict`): Search options for models that support it.
- `extra_body` (`dict`): Provider-specific request body overrides.

**`async run_stream(user_message: str | dict, ...) -> AsyncGenerator[str, None]`**

Stream the response token by token. Automatically handles tool calls in the background during the stream.

**`run_stream(user_message: str | dict, ...) -> Generator[str, None, None]`**

Sync streaming helper on `Agent` that yields chunks by driving the async stream
under the hood. This cannot be called from within an active event loop.

## LLM Client

### `iris_agent.BaseLLMClient`

OpenAI-compatible client wrapper.

```python
class BaseLLMClient:
    def __init__(self, config: LLMConfig) -> None
```

### `iris_agent.LLMConfig`

Configuration dataclass.

```python
@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: Optional[str] = None
    web_search_options: Optional[dict] = None
    extra_body: Optional[dict] = None
```

- `provider`: One of `LLMProvider` constants (e.g., `LLMProvider.OPENAI`).
- `model`: Model identifier (e.g., "gpt-4o").
- `base_url`: Optional override for compatible APIs.
- `extra_body`: Optional provider-specific request body overrides.

## Tools

### `iris_agent.tool`

Decorator to register a function as a tool.

```python
@tool(name="custom_name", description="Custom description")
def my_function(arg: int): ...
```

### `iris_agent.ToolRegistry`

Registry for managing tools.

#### Methods

**`register(func: Callable) -> ToolSpec`**

Register a single function.

**`register_from(obj: Any) -> None`**

Scan an object (or module) for methods decorated with `@tool` and register them.

## Prompts

### `iris_agent.PromptRegistry`

Registry for managing system prompts.

#### Methods

**`add_prompt(name: str, template: str | Callable)`**

Add a prompt.

**`render(prompt_name: str, **kwargs) -> str`**

Render a prompt by name, passing kwargs to `format` or the callable.

## Messages

### `iris_agent.create_message`

Helper to create standard message dictionaries.

```python
def create_message(
    role: str,
    content: str | None = None,
    name: str | None = None,
    images: List[str] | None = None
) -> dict
```

- `role`: "user", "assistant", "system", etc.
- `content`: Text content.
- `images`: List of image URLs.
