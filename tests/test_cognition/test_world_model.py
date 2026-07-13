from unittest.mock import MagicMock, patch

from iris_agent import LLMConfig, PromptRegistry, SyncLLMClient, ToolRegistry
from iris_agent.cognition import Mind, WorldModel, WorldModelAware


# ---------------------------------------------------------------------------
# WorldModel unit tests
# ---------------------------------------------------------------------------


def test_wm_init():
    wm = WorldModel()
    assert wm.graph is not None
    assert wm.current_goal_id == ""
    assert wm.session_id != ""


def test_wm_ingest_message():
    wm = WorldModel()
    mid = wm.ingest_message("user", "hello world")
    entity = wm.graph.get_entity(mid)
    assert entity is not None
    assert entity.type == "message"
    assert entity.properties["role"] == "user"
    assert entity.properties["content"] == "hello world"


def test_wm_set_goal():
    wm = WorldModel()
    gid = wm.set_goal("build a feature")
    assert wm.current_goal_id == gid
    entity = wm.graph.get_entity(gid)
    assert entity.properties["description"] == "build a feature"


def test_wm_add_task():
    wm = WorldModel()
    gid = wm.set_goal("main goal")
    tid = wm.add_task("sub task")
    assert tid != ""
    entity = wm.graph.get_entity(tid)
    assert entity.type == "task"
    assert entity.properties["description"] == "sub task"
    rels = wm.graph.get_relations(source_id=tid, rel_type="subgoal_of")
    assert len(rels) == 1
    assert rels[0].target_id == gid


def test_wm_add_task_no_goal():
    wm = WorldModel()
    tid = wm.add_task("orphan task")
    assert tid != ""
    rels = wm.graph.get_relations()
    assert len(rels) == 0


def test_wm_add_entity():
    wm = WorldModel()
    eid = wm.add_entity("file", {"name": "main.py"})
    entity = wm.graph.get_entity(eid)
    assert entity.type == "file"
    assert entity.properties["name"] == "main.py"


def test_wm_link():
    wm = WorldModel()
    a = wm.add_entity("goal", {"desc": "a"})
    b = wm.add_entity("task", {"desc": "b"})
    rid = wm.link(a, b, "requires")
    rel = wm.graph.get_relation(rid)
    assert rel.type == "requires"
    assert rel.source_id == a
    assert rel.target_id == b


def test_wm_get_related():
    wm = WorldModel()
    a = wm.add_entity("a", {})
    b = wm.add_entity("b", {})
    c = wm.add_entity("c", {})
    wm.link(a, b, "relates_to")
    wm.link(a, c, "relates_to")
    related = wm.get_related(a)
    assert len(related) == 2


def test_wm_get_context_empty():
    wm = WorldModel()
    ctx = wm.get_context()
    assert ctx == ""


def test_wm_get_context_with_goal():
    wm = WorldModel()
    wm.set_goal("finish project")
    ctx = wm.get_context()
    assert "Current Goal: finish project" in ctx


def test_wm_get_context_with_entities():
    wm = WorldModel()
    wm.set_goal("learn python")
    wm.add_entity("file", {"description": "main.py"})
    wm.add_entity("file", {"description": "utils.py"})
    ctx = wm.get_context()
    assert "Known entities:" in ctx
    assert "main.py" in ctx
    assert "utils.py" in ctx


def test_wm_ingest_tool_result():
    wm = WorldModel()
    eid = wm.ingest_tool_result("read_file", '{"path": "x.py"}', "file content")
    entity = wm.graph.get_entity(eid)
    assert entity.type == "tool_result"
    assert entity.properties["tool"] == "read_file"


def test_wm_record_belief():
    wm = WorldModel()
    wm.set_goal("test")
    bid = wm.record_belief("the sky is blue", 0.9)
    entity = wm.graph.get_entity(bid)
    assert entity.type == "belief"
    assert entity.properties["statement"] == "the sky is blue"
    assert entity.properties["confidence"] == 0.9
    rels = wm.graph.get_relations(source_id=bid)
    assert len(rels) == 1


def test_wm_record_lesson():
    wm = WorldModel()
    wm.set_goal("test")
    lid = wm.record_lesson("always validate input")
    entity = wm.graph.get_entity(lid)
    assert entity.type == "lesson"
    assert entity.properties["description"] == "always validate input"


def test_wm_serialize_roundtrip():
    wm = WorldModel()
    wm.set_goal("build")
    wm.ingest_message("user", "hi")
    wm.add_task("step 1")

    data = wm.to_dict()
    assert data["session_id"] == wm._session_id
    assert data["current_goal_id"] == wm._current_goal_id
    assert len(data["graph"]["entities"]) >= 3

    wm2 = WorldModel.from_dict(data)
    assert wm2.session_id == wm.session_id
    assert wm2.current_goal_id == wm.current_goal_id
    assert len(wm2.graph.find_entities()) == len(wm.graph.find_entities())
    assert (
        wm2.graph.get_entity(wm.current_goal_id).properties["description"]
        == "build"
    )


# ---------------------------------------------------------------------------
# WorldModelAware mixin
# ---------------------------------------------------------------------------


class FakeModule(WorldModelAware):
    pass


def test_world_model_aware_init():
    mod = FakeModule()
    assert mod.world_model is None


def test_world_model_aware_set():
    mod = FakeModule()
    wm = WorldModel()
    mod.set_world_model(wm)
    assert mod.world_model is wm


# ---------------------------------------------------------------------------
# Mind + WorldModel integration tests
# ---------------------------------------------------------------------------


def _make_mock_client():
    config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    client = SyncLLMClient(config)
    return client


def test_mind_world_model_created_by_default():
    client = _make_mock_client()
    mind = Mind(llm_client=client)
    assert mind.world_model is not None
    assert isinstance(mind.world_model, WorldModel)


def test_mind_world_model_disabled():
    client = _make_mock_client()
    mind = Mind(llm_client=client, use_world_model=False)
    assert mind.world_model is None


def test_mind_world_model_populated_on_run():
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    mind = Mind(llm_client=client, prompt_registry=prompts)

    with patch.object(client, "chat_completion", return_value={
        "content": "Hello",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Say hello")

    assert mind.world_model is not None
    entities = mind.world_model.graph.find_entities()
    types = {e.type for e in entities}
    assert "message" in types
    assert "goal" in types


def test_mind_world_model_context_in_prompt():
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    mind = Mind(llm_client=client, prompt_registry=prompts)

    with patch.object(client, "chat_completion", return_value={
        "content": "Hello",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Hello world")

    # The cognitive prompt should include World State
    cognitive_prompt = mind._build_cognitive_prompt({"selected_mental_model": ""})
    # Since world model is populated, get_context() should return non-empty
    wm_context = mind.world_model.get_context()
    assert "Current Goal: Hello world" in wm_context


def test_mind_world_model_disabled_has_no_wm():
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    mind = Mind(llm_client=client, prompt_registry=prompts, use_world_model=False)

    with patch.object(client, "chat_completion", return_value={
        "content": "Hello",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Say hello")

    assert mind.world_model is None
    # Original graph should still work
    assert len(mind.graph.all_nodes()) >= 1


def test_mind_world_model_with_tool_result():
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    tools = ToolRegistry()

    from iris_agent import tool

    @tool(description="Test tool")
    def test_tool(x: int) -> int:
        return x * 2

    tools.register(test_tool)

    mind = Mind(llm_client=client, prompt_registry=prompts, tool_registry=tools)

    # First call: triggers tool call
    # Second call: returns result
    response_with_tool_call = {
        "content": "",
        "tool_calls": [],
        "finish_reason": "tool_calls",
    }
    tool_mock = MagicMock(id="call_1")
    tool_mock.function.name = "test_tool"
    tool_mock.function.arguments = '{"x": 21}'
    response_with_tool_call["tool_calls"] = [tool_mock]

    call_responses = [
        response_with_tool_call,
        {
            "content": "The answer is 42",
            "tool_calls": [],
            "finish_reason": "stop",
        },
    ]

    call_iter = iter(call_responses)
    with patch.object(client, "chat_completion", side_effect=lambda **kwargs: next(call_iter)):
        result = mind.run("Compute 21 * 2")

    assert result.response == "The answer is 42"

    # World model should have the tool result entity
    tool_results = mind.world_model.graph.find_entities(
        entity_type="tool_result"
    )
    assert len(tool_results) >= 1
    assert tool_results[0].properties["tool"] == "test_tool"


def test_mind_world_model_records_lessons():
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    mind = Mind(llm_client=client, prompt_registry=prompts)

    with patch.object(client, "chat_completion", return_value={
        "content": "Done",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Task")

    lessons = mind.world_model.graph.find_entities(entity_type="lesson")
    # Reflector always produces at least one lesson
    assert len(lessons) >= 1


def test_mind_run_stream_with_world_model():
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
    # World model should still be populated
    assert mind.world_model is not None
    messages = mind.world_model.graph.find_entities(entity_type="message")
    assert len(messages) >= 1


# ---------------------------------------------------------------------------
# WorldModel persistence
# ---------------------------------------------------------------------------


def test_wm_save_and_load(tmp_path):
    wm = WorldModel()
    wm.set_goal("test persistence")
    wm.ingest_message("user", "hello")
    wm.add_task("step 1")

    path = str(tmp_path / "world_model.json")
    wm.save(path)

    loaded = WorldModel.load(path)
    assert loaded.current_goal_id == wm.current_goal_id
    assert len(loaded.graph.find_entities()) == len(wm.graph.find_entities())
    assert loaded.graph.get_entity(wm.current_goal_id).properties["description"] == "test persistence"


def test_wm_load_nonexistent():
    import pytest
    with pytest.raises(FileNotFoundError):
        WorldModel.load("/nonexistent/path.json")


# ---------------------------------------------------------------------------
# Mind world_model_path auto-save/load
# ---------------------------------------------------------------------------


def test_mind_world_model_path_autosaves(tmp_path):
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    wm_path = str(tmp_path / "test_wm.json")
    mind = Mind(llm_client=client, prompt_registry=prompts, world_model_path=wm_path)

    with patch.object(client, "chat_completion", return_value={
        "content": "Hello",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Hi")

    import os
    assert os.path.exists(wm_path)
    loaded = WorldModel.load(wm_path)
    assert loaded.current_goal_id != ""
    entities = loaded.graph.find_entities()
    assert any(e.type == "message" for e in entities)


def test_mind_world_model_path_autoloads(tmp_path):
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")

    wm_path = str(tmp_path / "test_wm.json")

    mind1 = Mind(llm_client=client, prompt_registry=prompts, world_model_path=wm_path)
    with patch.object(client, "chat_completion", return_value={
        "content": "Hello",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        mind1.run("First message")

    mind2 = Mind(llm_client=client, prompt_registry=prompts, world_model_path=wm_path)
    messages = mind2.world_model.graph.find_entities(entity_type="message")
    assert len(messages) >= 1
    assert any("First message" in m.properties.get("content", "") for m in messages)


def test_mind_world_model_path_disabled(tmp_path):
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    wm_path = str(tmp_path / "test_wm.json")

    mind = Mind(llm_client=client, prompt_registry=prompts, use_world_model=False, world_model_path=wm_path)
    assert mind.world_model is None

    with patch.object(client, "chat_completion", return_value={
        "content": "Hello",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        mind.run("Hi")

    import os
    # Should NOT save since world model is disabled
    assert not os.path.exists(wm_path)


# ---------------------------------------------------------------------------
# Cross-turn knowledge accumulation
# ---------------------------------------------------------------------------


def test_mind_cross_turn_knowledge(tmp_path):
    client = _make_mock_client()
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    wm_path = str(tmp_path / "cross_turn.json")

    mind = Mind(llm_client=client, prompt_registry=prompts, world_model_path=wm_path)

    with patch.object(client, "chat_completion", return_value={
        "content": "Hello back",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind.run("Turn 1: hello")

    assert mind.world_model is not None
    first_goal = mind.world_model.current_goal_id
    initial_entity_count = len(mind.world_model.graph.find_entities())

    # Second run — world model should carry over from file
    mind2 = Mind(llm_client=client, prompt_registry=prompts, world_model_path=wm_path)
    with patch.object(client, "chat_completion", return_value={
        "content": "Hello again",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        result = mind2.run("Turn 2: still here")

    assert mind2.world_model is not None
    # Should have more entities than the first run alone had
    second_count = len(mind2.world_model.graph.find_entities())
    assert second_count > initial_entity_count


def test_mind_world_model_observer_entity_extraction():
    """Observer automatically extracts file paths and creates entities."""
    client = _make_mock_client()
    mind = Mind(llm_client=client)

    with patch.object(client, "chat_completion", return_value={
        "content": "Done",
        "tool_calls": [],
        "finish_reason": "stop",
    }):
        mind.run("Read /Users/test/file.py and list src/")

    files = mind.world_model.graph.find_entities(entity_type="file")
    assert len(files) >= 1
    file_paths = {f.properties.get("path", "") for f in files}
    assert any("/Users/test/file.py" in p for p in file_paths) or any("src/" in p for p in file_paths)
