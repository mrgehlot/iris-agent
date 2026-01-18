# Modules Reference

The `iris_agent` package is structured as follows:

## `src/iris_agent/`

The core package source.

| Module | Description |
| :--- | :--- |
| **`agent.py`** | Contains the `Agent` class (Synchronous wrapper). |
| **`async_agent.py`** | Contains the `AsyncAgent` class (Core logic). |
| **`llm.py`** | Contains `BaseLLMClient`, `LLMConfig` and provider logic. |
| **`tools.py`** | Contains `ToolRegistry`, `@tool` decorator, and schema inference logic. |
| **`prompts.py`** | Contains `PromptRegistry` for managing system prompts. |
| **`messages.py`** | Helpers for creating messages (text, images). |
| **`types.py`** | Type definitions and constants (e.g., `Role`). |

## `examples/`

Contains reference implementations.

- **`01_basic/`**: Simple chat and streaming examples.
- **`02_prompts/`**: Using the prompt registry.
- **`03_tools/`**: Tool creation and usage.
- **`04_memory/`**: Memory inspection.
- **`05_multi_agent/`**: Patterns for multiple agents.
- **`09_gemini/`**: Specific examples for Google Gemini.

## `tests/`

Unit and integration tests using `pytest`.
