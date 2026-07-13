from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CognitiveEventType(str, Enum):
    OBSERVATION_CREATED = "observation_created"
    EXPERIENCE_STORED = "experience_stored"
    BELIEF_UPDATED = "belief_updated"
    CONFIDENCE_CHANGED = "confidence_changed"
    GOAL_ADDED = "goal_added"
    PLAN_CREATED = "plan_created"
    PLAN_CRITIQUED = "plan_critiqued"
    DECISION_MADE = "decision_made"
    REFLECTION_COMPLETED = "reflection_completed"
    LESSON_LEARNED = "lesson_learned"
    PREDICTION_FAILED = "prediction_failed"
    MENTAL_MODEL_SELECTED = "mental_model_selected"
    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_TESTED = "hypothesis_tested"


@dataclass
class CognitiveContext:
    input: str = ""
    processed_input: str = ""
    goal: str = ""
    reasoning_context: str = ""
    selected_mental_model: str = ""
    mental_model_explanation: str = ""
    plan: List[Dict[str, Any]] = field(default_factory=list)
    plan_critique: Optional[str] = None
    response: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class Observation:
    id: str
    content: str
    source: str = "user"
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experience:
    goal: str
    context: str
    action: str
    expected_outcome: str
    actual_outcome: str
    confidence: float = 0.0
    timestamp: float = 0.0
    reasoning_summary: str = ""
    lessons: List[str] = field(default_factory=list)
    mistakes: List[str] = field(default_factory=list)
    successful_strategies: List[str] = field(default_factory=list)
    id: str = ""


@dataclass
class Belief:
    statement: str
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    last_updated: float = 0.0
    id: str = ""


@dataclass
class PlanStep:
    action: str
    expected_outcome: str = ""
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None


@dataclass
class Decision:
    goal: str
    options: List[str]
    chosen: str
    reasoning: str
    confidence: float = 0.0
    timestamp: float = 0.0
    outcome: Optional[str] = None
    id: str = ""


@dataclass
class Hypothesis:
    statement: str
    confidence: float = 0.3
    predictions: List[str] = field(default_factory=list)
    status: str = "proposed"
    evidence: List[str] = field(default_factory=list)
    id: str = ""


@dataclass
class Critique:
    score: float
    issues: List[str]
    suggestions: List[str] = field(default_factory=list)


@dataclass
class Lesson:
    description: str
    category: str = "general"
    impact: float = 0.5
    applied_count: int = 0
    id: str = ""
