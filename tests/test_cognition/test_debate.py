from iris_agent.cognition import DebateEngine


def test_debate_process():
    engine = DebateEngine()
    context = {"input": "Should we use microservices?", "goal": "Evaluate architecture"}
    result = engine.process(context)
    assert "debate_result" in result
    assert result["debate_consensus"] != ""
    assert len(result["debate_result"].viewpoints) > 0


def test_debate_empty_input():
    engine = DebateEngine()
    result = engine.process({})
    assert "debate_result" not in result


def test_debate_default_roles():
    engine = DebateEngine()
    assert len(engine._roles) == 6
