from __future__ import annotations


def update_confidence(
    current: float,
    evidence_match: float,
    evidence_weight: float = 0.1,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    new_conf = current + evidence_weight * (evidence_match - current)
    return max(min_value, min(max_value, new_conf))


def confidence_from_repetition(
    success_count: int,
    total_count: int,
    prior: float = 0.5,
    prior_weight: int = 2,
) -> float:
    return (prior * prior_weight + success_count) / (prior_weight + total_count)


def confidence_after_reflection(
    prior_confidence: float,
    lessons_learned: int,
    mistakes_made: int,
    strategies_found: int,
    lesson_weight: float = 0.05,
    mistake_penalty: float = 0.1,
    strategy_bonus: float = 0.08,
) -> float:
    delta = (
        lessons_learned * lesson_weight
        - mistakes_made * mistake_penalty
        + strategies_found * strategy_bonus
    )
    return max(0.0, min(1.0, prior_confidence + delta))
