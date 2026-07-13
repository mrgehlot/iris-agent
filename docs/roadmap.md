# Iris Agent Framework — Evolution Roadmap

Transform the lightweight LLM orchestration library into a modular Cognitive
Architecture Framework inspired by Marvin Minsky's *Society of Mind*.

**Guiding principle:** The framework should model cognition, not prompts. The
LLM is one reasoning component among many, coordinated by explicit cognitive
modules.

---

## Design Decisions

These decisions govern the entire effort and will not be re-litigated module by
module.

| Decision | Choice | Rationale |
|---|---|---|
| Mind vs Agent | Mind replaces Agent for cognitive use. `Agent`/`AsyncAgent` remain untouched for backward compat. | Users who just want LLM + tools keep using `Agent`. Users who want cognition use `Mind`. |
| Sync/Async pattern | Two classes: `Mind` (sync) and `AsyncMind` (async), mirroring the existing `Agent`/`AsyncAgent` pattern. | Consistent with the existing API surface. No single-class dual-mode complexity. |
| Module communication | Mind orchestrates modules by calling their `process()` directly in a pipeline. No magic message bus. | Simple, debuggable, traceable. Explicit pipeline order. |
| Mental model injection | The Thinker automatically wraps mental model output into the system prompt. Users never manually manage reasoning prompts. | The whole point of cognitive modules is to make reasoning visible to the LLM without boilerplate. |
| LLM access from modules | Modules that need LLM access (Critic, Reflector, Debate) receive a reference to the LLM client. They call it directly. | Modules are self-contained. No hidden magic. |
| Event system | Synchronous pub/sub for lifecycle hooks. In-process only. No queues. | Simple, zero-dependency. Async events can be added in Phase 5 if needed. |
| Experience graph | In-memory with abstract `GraphStore` interface. File/DB implementations deferred to Phase 5. | Phase 1 focuses on the cognitive model. Persistence is infrastructure. |
| Confidence model | Simple Bayesian-ish update: `new_c = c + evidence * (match - c)`. Not full probability theory. | Good enough for Phase 1. Can be swapped for a proper Bayesian engine later. |
| Phase 1 module depth | All 16+ modules exist but some are thin scaffolds (Simulation, Debate, Learning are basic implementations). | Build the architecture first, enrich later. |

---

## Phase 1 — Cognitive Core

Build the foundational cognitive modules and the `Mind`/`AsyncMind` orchestrators.

### Package Structure

```
src/iris_agent/
  __init__.py              ← add cognition- module exports
  agent.py                 ← unchanged
  async_agent.py           ← unchanged
  _utils.py                ← unchanged
  cognition/
    __init__.py
    base.py                ← CognitiveModule ABC
    events.py              ← Sync event bus
    types.py               ← Shared dataclasses (CognitiveContext, etc.)
    mind.py                ← Mind orchestrator (sync)
    async_mind.py          ← AsyncMind orchestrator

    # Data layer
    experiences.py         ← Experience dataclass, ExperienceStore
    beliefs.py             ← Belief dataclass, BeliefSystem
    confidence.py          ← Confidence scoring + update
    graph.py               ← Cognition graph + abstract store interface

    # Perception
    observer.py            ← Input → Observation → Experience

    # Reasoning
    mental_models.py       ← MentalModel ABC + built-in library
    thinker.py             ← Selects and invokes mental models
    planner.py             ← Goal decomposition

    # Evaluation
    critic.py              ← Self-evaluation of plans + outputs
    reflector.py           ← Post-action analysis → lessons → belief updates

    # Synthesis
    hypothesis.py          ← Hypothesis generation + testing
    decisions.py           ← Decision records + log
    attention.py           ← Focus / priority management

    # Advanced (scaffolds in Phase 1)
    debate.py              ← Multi-perspective reasoning
    simulation.py          ← Counterfactual simulation
    learning.py            ← Strategy adaptation from experience
```

### Build Order

1. **Package skeleton** — `__init__.py`, `base.py`, `types.py`, `events.py`
2. **Data layer** — `experiences.py`, `beliefs.py`, `confidence.py`, `graph.py`
3. **Reasoning modules** — `mental_models.py`, `thinker.py`, `planner.py`
4. **Perception + evaluation** — `observer.py`, `critic.py`, `reflector.py`
5. **Support modules** — `hypothesis.py`, `decisions.py`, `attention.py`
6. **Advanced scaffolds** — `debate.py`, `simulation.py`, `learning.py`
7. **Orchestration** — `mind.py`, `async_mind.py`
8. **Integration** — update `__init__.py`, write tests

### Cognitive Pipeline

```
User Input
    │
    ▼
┌─────────────┐
│  Observer    │  Creates Observation → Experience (goal, context)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Thinker     │  Selects MentalModel(s) by applicability()
│  ┌─────────┐│  Runs top models, wraps reasoning into prompt
│  │Mental   ││
│  │Models   ││
│  └─────────┘│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Planner     │  Decomposes goal → PlanSteps
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Critic      │  Evaluates plan. Low score → loop back to Planner
└──────┬──────┘
       │ (plan approved)
       ▼
┌─────────────────────────────────────────┐
│  Execute plan via LLM + ToolRegistry    │
│  (same loop as Agent.run)               │
└──────┬──────────────────────────────────┘
       │ (each tool result / response)
       ▼
┌─────────────┐
│  Reflector   │  Creates Experience from execution
│              │  Extracts: Lessons, Mistakes, Strategies
└──────┬──────┘  Updates: Beliefs, Confidence
       │
       ▼
┌─────────────┐
│  Beliefs     │  Update belief confidences with new evidence
│  Confidence  │
│  Graph       │  Add cognition nodes + edges
└─────────────┘
       │
       ▼
┌─────────────┐
│  Decisions   │  Record the final Decision with full trace
└─────────────┘
       │
       ▼
   Output (response + confidence + decisions + graph snapshot)
```

### Backward Compatibility

```python
# Old API — unchanged, still works
from iris_agent import Agent
agent = Agent(llm_client=client)
agent.run("hello")

# New API — opt-in via Mind
from iris_agent.cognition import Mind
mind = Mind(llm_client=client, tool_registry=tools)
result = mind.run("hello")
result.response       # str — same as Agent.run()
result.confidence     # float
result.decisions      # list[Decision]
result.graph          # cognition graph node IDs
```

### Testing Strategy

- Every cognitive module has its own test file under `tests/test_cognition/`
- Modules that don't need LLM access are fully tested without API keys
- Modules that need LLM access (Critic, Reflector) accept mock LLM clients
- `Mind` is tested with mocked cognitive modules for integration
- All existing `test_basic.py` tests continue to pass unchanged

---

## Phase 2 — World Model

Replace flat message history with a structured internal representation of the
environment: goals, projects, files, tasks, tools, constraints, and their
relationships. Reasoning operates over the world model instead of raw chat
messages.

**Key modules:** `world_model.py`, `WorldModelAware` mixin for modules.

---

## Phase 3 — Internal Debate & Simulation

Replace single-pass reasoning with multi-role debate and counterfactual
evaluation. The planner produces candidates, debate evaluates them, critic
scores, and the winner is executed.

**Key upgrades:** `debate.py` (full role system), `simulation.py` (full
counterfactual engine).

---

## Phase 4 — Learning & Adaptation

Meta-cognitive adaptation: automatic mental model selection based on historical
success rates, confidence calibration, experience compression, and periodic
self-reflection on reasoning quality.

**Key upgrades:** `learning.py` (full strategy adaptation), `reflector.py`
(meta-cognitive pass).

---

## Phase 5 — Persistence & Extensibility

Production-hardening: file/DB stores for graphs and beliefs, plugin system for
external `CognitiveModule` subclasses, REST API for cognitive state inspection,
OpenTelemetry integration.

**Key additions:** `GraphStore` implementations (JSON, SQLite), plugin
discovery, REST + WebSocket server.
