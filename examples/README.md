## Iris Agent Examples

These examples show common ways to use `iris-agent`:

- Basic agent usage (sync, async, streaming)
- Prompt registry (static and dynamic prompts)
- Tool registry (sync and async tools, custom schemas)
- Memory usage (reading, seeding, and clearing)
- Multiple agents collaborating
- Custom LLM client (mock client for local testing)
- Rich logging
- Message creation helpers
- Gemini models via OpenAI-compatible base URL

### Requirements

Most examples require:

- `OPENAI_API_KEY`
- Optional: `OPENAI_MODEL` (defaults are set in examples)
- Optional: `OPENAI_BASE_URL`

### Run an example

From the repo root:

```bash
python examples/01_basic/simple_agent.py
```

Each file is standalone and can be executed directly.
