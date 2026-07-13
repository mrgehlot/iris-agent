"""
Medium-hard test for the Mind + WorldModel pipeline.

Exercises: Observer → Thinker → Planner → Critic → LLM+Tools → Reflector → WorldModel
"""

import os

from iris_agent import LLMConfig, SyncLLMClient, ToolRegistry
from iris_agent.cognition import Mind

# 1. LLM client
config = LLMConfig(
    provider="openai",
    model="gpt-5-nano",
    api_key=os.getenv("OPENAI_API_KEY"),
)
client = SyncLLMClient(config)

# 2. Tools — include core built-ins
tools = ToolRegistry()
tools.include_core()

# 3. Mind with WorldModel
mind = Mind(
    llm_client=client,
    tool_registry=tools,
    system_prompt_name="assistant",
    enable_logging=True
)

# 4. Run
question = """
I have a Python project at /Users/mrgehlot/Desktop/coding/iris-agent.
1. List all .py files in src/iris_agent/cognition/ and count them.
2. Search for any function whose name or docstring mentions 'world_model'.
3. Summarize your findings: how many cognitive modules exist,
   which ones reference the world model, and give me a confidence score
   for your summary.
"""
result = mind.run(question)

print("\n" + "=" * 60)
print("RESPONSE:")
print(result.response)
print("=" * 60)
print(f"\nConfidence: {result.confidence}")
print(f"Mental model: {result.mental_model}")
print(f"Lessons learned: {result.lessons}")
print(f"Decisions: {result.decisions}")

# 5. Inspect WorldModel
print("\n" + "=" * 60)
print("WORLD MODEL STATE:")
for entity in mind.world_model.graph.find_entities():
    desc = entity.properties.get("description") or entity.properties.get("content", "")
    print(f"  [{entity.type}] {desc[:80]}")
print(f"\nTotal entities: {mind.world_model.graph.stats()['total_entities']}")
