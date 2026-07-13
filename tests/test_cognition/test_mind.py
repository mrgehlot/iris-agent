from unittest.mock import MagicMock, patch

from iris_agent import LLMConfig, PromptRegistry, SyncLLMClient, ToolRegistry
from iris_agent.cognition import Mind


def _make_mock_client():
    config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    client = SyncLLMClient(config)
    return client


def test_mind_init():
    client = _make_mock_client()
    mind = Mind(llm_client=client)
    assert mind.observer is not None
    assert mind.thinker is not None
    assert mind.planner is not None
    assert mind.critic is not None
    assert mind.reflector is not None
    assert mind.experience_store is not None
    assert mind.belief_system is not None
    assert mind.decision_log is not None
    assert mind.graph is not None


def test_mind_run_with_mock():
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    mind = Mind(llm_client=client, prompt_registry=prompts)

    with patch.object(client, "chat_completion", return_value={
        "content": "Hello from mock LLM",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Say hello")

    assert result.response == "Hello from mock LLM"
    assert result.confidence >= 0.0
    assert len(result.graph_node_ids) > 0


def test_mind_with_tools():
    client = _make_mock_client()
    tools = ToolRegistry()

    from iris_agent import tool

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    tools.register(add)

    mind = Mind(llm_client=client, tool_registry=tools)

    with patch.object(client, "chat_completion", return_value={
        "content": "The sum is 5",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("What is 2 + 3?")

    assert result.response == "The sum is 5"


def test_mind_debate_enabled():
    client = _make_mock_client()
    mind = Mind(llm_client=client, enable_debate=True)
    assert mind._debate is not None


def test_mind_simulation_enabled():
    client = _make_mock_client()
    mind = Mind(llm_client=client, enable_simulation=True)
    assert mind._simulation is not None


def test_mind_run_stream():
    client = _make_mock_client()
    mind = Mind(llm_client=client)

    class FakeChunk:
        def __init__(self, text):
            self.choices = [MagicMock()]
            self.choices[0].delta.content = text
            self.choices[0].delta.tool_calls = None

    def fake_stream(*args, **kwargs):
        yield FakeChunk("Hello ")
        yield FakeChunk("world!")

    with patch.object(client, "chat_completion_stream", fake_stream):
        result = list(mind.run_stream("Hi"))

    assert "".join(result) == "Hello world!"


def test_mind_graph_populated():
    client = _make_mock_client()
    mind = Mind(llm_client=client)

    with patch.object(client, "chat_completion", return_value={
        "content": "Response",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Test")

    assert len(mind.graph.all_nodes()) >= 2
    assert len(mind.graph.all_edges()) >= 1


def test_mind_experience_created():
    client = _make_mock_client()
    mind = Mind(llm_client=client)

    with patch.object(client, "chat_completion", return_value={
        "content": "Response",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Test")

    exps = mind.experience_store.all()
    assert len(exps) >= 1
