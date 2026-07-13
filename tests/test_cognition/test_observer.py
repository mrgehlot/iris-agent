from iris_agent.cognition import Observer, WorldModel


def test_observer_processes_input():
    observer = Observer()
    context = {"input": "Hello world", "goal": "Hello world"}
    result = observer.process(context)
    assert "observation_id" in result
    assert result["processed_input"] == "Hello world"
    assert result["goal"] == "Hello world"


def test_observer_no_input():
    observer = Observer()
    context = {}
    result = observer.process(context)
    assert "observation_id" in result
    assert result.get("processed_input") == ""


def test_observer_entity_extraction_file_paths():
    wm = WorldModel()
    observer = Observer()
    observer.set_world_model(wm)

    observer.process({"input": "Read /home/user/file.py and check ./data/results.csv", "goal": "test"})

    files = wm.graph.find_entities(entity_type="file")
    assert len(files) >= 2
    paths = [e.properties.get("path", "") for e in files]
    assert any("/home/user/file.py" in p for p in paths)


def test_observer_entity_extraction_urls():
    wm = WorldModel()
    observer = Observer()
    observer.set_world_model(wm)

    observer.process({"input": "Fetch https://example.com/api and check http://test.dev", "goal": "test"})

    urls = wm.graph.find_entities(entity_type="url")
    assert len(urls) >= 2


def test_observer_entity_extraction_tool_mentions():
    wm = WorldModel()
    observer = Observer()
    observer.set_world_model(wm)

    observer.process({"input": "Use read_file on that file then glob_files for *.py", "goal": "test"})

    tools = wm.graph.find_entities(entity_type="tool_mention")
    names = {e.properties.get("name", "") for e in tools}
    assert "read_file" in names
    assert "glob_files" in names


def test_observer_world_model_sets_goal():
    wm = WorldModel()
    observer = Observer()
    observer.set_world_model(wm)

    observer.process({"input": "Help me refactor the code", "goal": "refactor the code"})

    assert wm.current_goal_id != ""
    goal = wm.graph.get_entity(wm.current_goal_id)
    assert goal.properties.get("description") == "refactor the code"


def test_observer_world_model_ingests_message():
    wm = WorldModel()
    observer = Observer()
    observer.set_world_model(wm)

    observer.process({"input": "Hello world", "goal": "test"})

    messages = wm.graph.find_entities(entity_type="message")
    assert len(messages) >= 1
    assert any("Hello world" in m.properties.get("content", "") for m in messages)
