# Frequently Asked Questions

## General

### Does Iris Agent support local LLMs?
**Yes.** You can use any OpenAI-compatible API (Ollama, vLLM, LocalAI) by setting the `base_url` in `LLMConfig`.

### Is conversation history persisted to disk?
**No.** By default, memory is stored in a Python list (`agent.memory`). If you restart the script, memory is lost. You can easily implement persistence by saving/loading `agent.memory` to a JSON file or database.

## Technical

### How do I clear the agent's memory?
Simply assign a new list or call clear on the existing one, but remember to keep the system prompt if you want it to persist.

```python
# Keep system prompt (index 0)
agent.memory = [agent.memory[0]]
```

### Can I share tools between agents?
**Yes.** You can pass the same `ToolRegistry` instance to multiple agents.

### Is it thread-safe?
The `AsyncAgent` is designed for `asyncio` concurrency. The synchronous `Agent` is not thread-safe if shared across threads without locks, as it modifies internal state (memory). It is better to create one agent per session/user.
