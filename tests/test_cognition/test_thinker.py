from iris_agent.cognition import Thinker


def test_thinker_process_with_reasoning():
    thinker = Thinker(threshold=0.0)
    context = {"input": "Test input", "goal": "Test goal"}
    result = thinker.process(context)
    assert "reasoning_context" in result
    assert result["reasoning_context"] != ""
    assert result["selected_mental_model"] != ""


def test_thinker_with_high_threshold():
    thinker = Thinker(threshold=0.9)
    context = {"input": "Test"}
    result = thinker.process(context)
    assert result["selected_mental_model"] == "none"


def test_thinker_max_models():
    thinker = Thinker(threshold=0.0, max_models=1)
    context = {"input": "Test"}
    result = thinker.process(context)
    assert "reasoning_context" in result


def test_thinker_empty_input():
    thinker = Thinker(threshold=0.0)
    context = {}
    result = thinker.process(context)
    assert "selected_mental_model" in result
