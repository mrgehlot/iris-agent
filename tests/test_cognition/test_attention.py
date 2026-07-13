from iris_agent.cognition import Attention


def test_attention_creates_summary():
    attention = Attention()
    context = {
        "input": "Do something important",
        "goal": "Complete task",
        "reasoning_context": "First principles analysis",
    }
    result = attention.process(context)
    assert "attention_summary" in result
    assert "Goal:" in result["attention_summary"]


def test_attention_empty():
    attention = Attention()
    result = attention.process({})
    assert result.get("attention_summary") == ""


def test_attention_max_items():
    attention = Attention(max_context_items=3)
    context = {"input": "test" * 100, "goal": "goal"}
    result = attention.process(context)
    assert "focused_input" in result
