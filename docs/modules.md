# Modules Reference

The `iris_agent` package is structured as follows:

## `src/iris_agent/`

The core package source.

| Module | Description |
| :--- | :--- |
| **`agent.py`** | Contains the `Agent` class (Synchronous agent). |
| **`async_agent.py`** | Contains the `AsyncAgent` class (Core logic). |
| **`llm.py`** | Contains `BaseLLMClient`, `SyncLLMClient`, `AsyncLLMClient`, and provider logic. |
| **`tools.py`** | Contains `ToolRegistry`, `@tool` decorator, and schema inference logic. |
| **`prompts.py`** | Contains `PromptRegistry` for managing system prompts. |
| **`messages.py`** | Helpers for creating messages (text, images). |
| **`types.py`** | Type definitions and constants (e.g., `Role`). |

## `src/iris_agent/cognition/` (Phase 1 — Cognitive Architecture)

The cognitive package (opt-in, import from `iris_agent.cognition`):

| Module | Description |
| :--- | :--- |
| **`mind.py`** | `Mind` — synchronous orchestrator for the cognitive pipeline |
| **`async_mind.py`** | `AsyncMind` — asynchronous orchestrator |
| **`observer.py`** | `Observer` — input → observation → experience |
| **`thinker.py`** | `Thinker` — mental model selection and execution |
| **`mental_models.py`** | `MentalModel` ABC + 6 built-in models (First Principles, RCA, OODA, SWOT, Second-Order Thinking, Bayesian) |
| **`planner.py`** | `Planner` — goal decomposition into plan steps |
| **`critic.py`** | `Critic` — self-evaluation and revision |
| **`reflector.py`** | `Reflector` — post-action analysis and lesson recording |
| **`attention.py`** | `AttentionController` — focus/priority management |
| **`debate.py`** | `DebateModule` — multi-perspective reasoning |
| **`simulation.py`** | `SimulationModule` — counterfactual evaluation |
| **`learning.py`** | `LearningModule` — strategy adaptation |
| **`hypothesis.py`** | `HypothesisGenerator` — hypothesis generation and testing |
| **`beliefs.py`** | `BeliefSystem` — probabilistic beliefs with evidence tracking |
| **`confidence.py`** | `ConfidenceTracker` — confidence estimation |
| **`decisions.py`** | `DecisionLog` — decision recording |
| **`events.py`** | `EventBus` — simplified pub/sub for in-pipeline events |
| **`experiences.py`** | `ExperienceStore` — experience creation and storage |
| **`graph.py`** | `KnowledgeGraph` — Entity/Relation CRUD, traversal, serialization |
| **`world_model.py`** | `WorldModel` — facade over KnowledgeGraph with high-level API |
| **`world_model_aware.py`** | `WorldModelAware` — mixin for modules that need world model access |
| **`base.py`** | `CognitiveModule` ABC — all modules inherit this |

## `examples/`

Contains reference implementations.

- **`01_basic/`**: Simple chat and streaming examples.
- **`02_prompts/`**: Using the prompt registry.
- **`03_tools/`**: Tool creation and usage.
- **`04_memory/`**: Memory inspection.
- **`05_multi_agent/`**: Patterns for multiple agents.
- **`09_gemini/`**: Specific examples for Google Gemini.

## `tests/`

Unit and integration tests using `pytest`.
