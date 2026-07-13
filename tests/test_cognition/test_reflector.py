from iris_agent.cognition import Reflector, WorldModel


def test_reflector_without_action():
    reflector = Reflector()
    context = {"goal": "Test"}
    result = reflector.process(context)
    assert result.get("reflection_completed") is None


def test_reflector_with_response():
    reflector = Reflector()
    context = {
        "goal": "Test goal",
        "last_action": "llm_response",
        "response": "Completed",
        "expected_outcome": "Completed",
    }
    result = reflector.process(context)
    assert result["reflection_completed"] is True
    assert len(result["lessons"]) > 0


def test_reflector_mismatch():
    reflector = Reflector()
    context = {
        "goal": "Test goal",
        "last_action": "llm_response",
        "response": "Failed",
        "expected_outcome": "Succeeded",
    }
    result = reflector.process(context)
    assert len(result["mistakes"]) > 0


def test_reflector_records_lessons_in_world_model():
    wm = WorldModel()
    wm.set_goal("test goal")
    reflector = Reflector()
    reflector.set_world_model(wm)

    context = {
        "goal": "test goal",
        "last_action": "llm_response",
        "response": "Completed",
        "expected_outcome": "Completed",
    }
    result = reflector.process(context)

    lessons = wm.graph.find_entities(entity_type="lesson")
    assert len(lessons) >= 1
    assert any("Completed goal successfully" in l.properties.get("description", "") for l in lessons)


def test_reflector_world_model_no_action_skips_recording():
    wm = WorldModel()
    reflector = Reflector()
    reflector.set_world_model(wm)

    context = {"goal": "test"}
    result = reflector.process(context)

    lessons = wm.graph.find_entities(entity_type="lesson")
    assert len(lessons) == 0
