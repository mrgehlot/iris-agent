# Examples

The `iris-agent` repository contains a rich set of examples in the `examples/` directory.

## Featured: Multi-Agent Pattern

Here is a complete example of a **Planner/Executor** pattern, where one agent creates a plan and another executes it.

```python
import os
from iris_agent import Agent, BaseLLMClient, LLMConfig, LLMProvider, PromptRegistry

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY first.")
        return

    # 1. Setup Configuration
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        api_key=api_key
    )
    client = BaseLLMClient(config)
    prompts = PromptRegistry()

    # 2. Define Prompts
    prompts.add_prompt("planner", "You are a strategic planner. Create a 3-step plan.")
    prompts.add_prompt("executor", "You are a doer. Execute the plan provided.")

    # 3. Initialize Agents
    planner = Agent(llm_client=client, prompt_registry=prompts, system_prompt_name="planner")
    executor = Agent(llm_client=client, prompt_registry=prompts, system_prompt_name="executor")

    # 4. Run Workflow
    task = "Write a haiku about Python."
    
    print("--- Planning ---")
    plan = planner.run(f"Create a plan for: {task}")
    print(plan)

    print("\n--- Executing ---")
    final_result = executor.run(f"Original Task: {task}\n\nPlan:\n{plan}")
    print(final_result)

if __name__ == "__main__":
    main()
```

## Directory Structure

You can find more specific examples in the `examples/` folder of the repository:

### Basic Usage (`examples/01_basic/`)
- **`simple_agent.py`**: A minimal synchronous agent.
- **`async_agent.py`**: The same agent using `asyncio`.
- **`streaming_agent.py`**: How to stream tokens in real-time.
- **`sync_streaming_agent.py`**: Sync streaming with `Agent.run_stream`.

### Prompts (`examples/02_prompts/`)
- **`prompt_registry_basic.py`**: Registering and using static prompts.
- **`prompt_registry_dynamic.py`**: Using templates (e.g., `{name}`) in system prompts.

### Tools (`examples/03_tools/`)
- **`tool_registry_basic.py`**: Registering simple functions.
- **`async_tools.py`**: Tools that perform async operations (e.g., `async def fetch_url(...)`).
- **`tool_schema_custom.py`**: Advanced type hinting for tools.

### Memory (`examples/04_memory/`)
- **`memory_basics.py`**: Inspecting and manipulating `agent.memory`.

### Multi-Agent (`examples/05_multi_agent/`)
- **`two_agents_chat.py`**: Two agents talking to each other.
- **`planner_executor.py`**: One agent creates a plan, another executes it (Orchestrator pattern).

### Custom LLM (`examples/06_custom_llm/`)
- **`mock_llm_client.py`**: How to implement `BaseLLMClient` for testing or custom providers.

### Logging (`examples/07_logging/`)
- **`rich_logging.py`**: Demonstration of the beautiful terminal output.

### Messages (`examples/08_messages/`)
- **`create_message_examples.py`**: Creating user, assistant, and multimodal (image) messages.

### Gemini (`examples/09_gemini/`)
- **`gemini_basic.py`**: Connecting to Google's Gemini models.
