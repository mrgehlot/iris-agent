from iris_agent.cognition import (
    confidence_after_reflection,
    confidence_from_repetition,
    update_confidence,
)


def test_update_confidence_increases():
    result = update_confidence(0.5, evidence_match=0.8, evidence_weight=0.1)
    assert 0.5 < result < 0.8


def test_update_confidence_decreases():
    result = update_confidence(0.5, evidence_match=0.2, evidence_weight=0.1)
    assert 0.2 < result < 0.5


def test_update_confidence_clamps():
    result = update_confidence(0.5, evidence_match=2.0, evidence_weight=1.0)
    assert result == 1.0


def test_confidence_from_repetition():
    result = confidence_from_repetition(success_count=8, total_count=10)
    assert 0.7 < result < 0.9


def test_confidence_from_repetition_all_fail():
    result = confidence_from_repetition(success_count=0, total_count=10)
    assert result < 0.2


def test_confidence_after_reflection():
    result = confidence_after_reflection(
        prior_confidence=0.5,
        lessons_learned=2,
        mistakes_made=1,
        strategies_found=1,
    )
    assert 0.48 < result < 0.6


def test_confidence_after_reflection_no_mistakes():
    result = confidence_after_reflection(
        prior_confidence=0.5,
        lessons_learned=3,
        mistakes_made=0,
        strategies_found=2,
    )
    assert result > 0.6
