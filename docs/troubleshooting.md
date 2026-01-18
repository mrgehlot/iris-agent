# Troubleshooting

## Common Issues

### 1. "RuntimeError: Agent.run called inside an event loop"

**Cause**: You are using the synchronous `Agent` class inside an async environment (like a Jupyter Notebook or a FastAPI route).

**Solution**: Use `AsyncAgent` instead.

```python
# Bad
agent = Agent(...)
agent.run("hi")

# Good
agent = AsyncAgent(...)
await agent.run("hi")
```

### 2. Tools are not being called

**Cause**:
- The tool might not be registered with the `ToolRegistry`.
- The `ToolRegistry` was not passed to the `Agent` constructor.
- The tool's docstring description is too vague for the LLM to understand when to use it.

**Solution**:
- Check registration: `print(registry.list_tools())`.
- Improve docstrings: Explain *what* the tool does and *when* to use it.

### 3. API Connection Errors

**Cause**: Missing API key or incorrect Base URL.

**Solution**:
- Ensure `OPENAI_API_KEY` (or equivalent) is set.
- If using a local model (e.g., Ollama), ensure `base_url` is correct (e.g., `http://localhost:11434/v1`).

### 4. Import Errors

**Cause**: The package is not installed in the current environment.

**Solution**: Run `pip install .` or `pip install -e .` in the project root.
