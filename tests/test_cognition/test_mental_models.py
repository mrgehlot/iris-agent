from iris_agent.cognition import (
    FirstPrinciples,
    MentalModelRegistry,
    RootCauseAnalysis,
    create_default_registry,
)


def test_first_principles():
    model = FirstPrinciples()
    assert model.name == "first_principles"
    assert model.applicability({}) == 0.7
    result = model.reason({"goal": "Test goal"})
    assert result.model_name == "first_principles"
    assert len(result.reasoning) > 0


def test_root_cause():
    model = RootCauseAnalysis()
    assert model.name == "root_cause_analysis"
    result = model.reason({"goal": "Find bug"})
    assert "root cause" in result.reasoning.lower()


def test_default_registry():
    registry = create_default_registry()
    assert len(registry.all()) >= 6


def test_candidates():
    registry = create_default_registry()
    candidates = registry.candidates({"goal": "test"}, threshold=0.5)
    assert len(candidates) > 0


def test_candidates_with_threshold():
    registry = create_default_registry()
    candidates = registry.candidates({"goal": "test"}, threshold=0.8)
    assert len(candidates) == 0


def test_register_and_get():
    registry = MentalModelRegistry()
    model = FirstPrinciples()
    registry.register(model)
    assert registry.get("first_principles") is model
    assert registry.get("nonexistent") is None
