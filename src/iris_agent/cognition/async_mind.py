from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from .._utils import safe_json_loads, setup_agent_logger, truncate
from ..llm import AsyncLLMClient
from ..messages import create_message
from ..prompts import PromptRegistry
from ..tools import ToolRegistry
from ..types import Role as MessageRole
from .attention import Attention
from .base import CognitiveModule, ModuleRegistry
from .beliefs import BeliefSystem
from .critic import Critic
from .debate import DebateEngine
from .decisions import DecisionLog
from .events import CognitiveEventType, EventBus
from .experiences import ExperienceStore
from .graph import CognitionEdge, CognitionNode, DictGraphStore
from .hypothesis import HypothesisGenerator
from .learning import Learning
from .mental_models import MentalModelRegistry
from .observer import Observer
from .planner import Planner
from .reflector import Reflector
from .simulation import Simulation
from .thinker import Thinker
from .world_model import WorldModel
from .world_model_aware import WorldModelAware


class AsyncMind:
    def __init__(
        self,
        llm_client: AsyncLLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        system_prompt_name: str = "assistant",
        mental_model_registry: Optional[MentalModelRegistry] = None,
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
        enable_debate: bool = False,
        enable_simulation: bool = False,
        use_world_model: bool = True,
        world_model_path: Optional[str] = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry or ToolRegistry()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.system_prompt_name = system_prompt_name
        self.logger = logger or (setup_agent_logger() if enable_logging else None)

        self._memory: List[dict] = []
        self.use_world_model = use_world_model
        self.world_model_path = world_model_path

        self.event_bus = EventBus()
        self.modules = ModuleRegistry()
        self.experience_store = ExperienceStore()
        self.belief_system = BeliefSystem()
        self.decision_log = DecisionLog()
        self.graph = DictGraphStore()
        if use_world_model:
            self.world_model = self._load_world_model() if world_model_path else WorldModel()
        else:
            self.world_model = None

        self.observer = Observer()
        self.thinker = Thinker(registry=mental_model_registry)
        self.planner = Planner()
        self.critic = Critic()
        self.reflector = Reflector()
        self.attention = Attention()
        self.hypothesis_generator = HypothesisGenerator()
        self.learning = Learning()

        self._debate = DebateEngine() if enable_debate else None
        self._simulation = Simulation() if enable_simulation else None

        for name, module in [
            ("observer", self.observer),
            ("thinker", self.thinker),
            ("planner", self.planner),
            ("critic", self.critic),
            ("reflector", self.reflector),
            ("attention", self.attention),
            ("hypothesis_generator", self.hypothesis_generator),
            ("learning", self.learning),
        ]:
            self.modules.register(name, module)
            module.attach(self)

        if self._debate:
            self.modules.register("debate", self._debate)
            self._debate.attach(self)
        if self._simulation:
            self.modules.register("simulation", self._simulation)
            self._simulation.attach(self)

        self._inject_world_model()

    def _load_world_model(self) -> WorldModel:
        if self.world_model_path and os.path.exists(self.world_model_path):
            try:
                return WorldModel.load(self.world_model_path)
            except Exception:
                pass
        return WorldModel()

    def _save_world_model(self) -> None:
        if self.world_model and self.world_model_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.world_model_path)), exist_ok=True)
                self.world_model.save(self.world_model_path)
            except Exception:
                pass

    def _inject_world_model(self) -> None:
        if not self.use_world_model or self.world_model is None:
            return
        for module in self.modules.all():
            if isinstance(module, WorldModelAware):
                module.set_world_model(self.world_model)

    async def run(
        self,
        user_message: str | dict,
        json_response: bool = False,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
        extra_body: dict | None = None,
    ) -> MindOutput:
        context: Dict[str, Any] = {
            "input": user_message if isinstance(user_message, str) else str(user_message),
            "goal": user_message if isinstance(user_message, str) else str(user_message),
            "json_response": json_response,
            "max_completion_tokens": max_completion_tokens,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "web_search_options": web_search_options,
            "extra_body": extra_body,
        }

        context = self.observer.process(context)
        context = self.attention.process(context)
        context = self.thinker.process(context)
        context = self.planner.process(context)
        context = self.critic.process(context)

        if context.get("plan_revision_needed"):
            context = self.planner.process(context)
            context = self.critic.process(context)

        if self._debate:
            context = self._debate.process(context)

        if self._simulation:
            context = self._simulation.process(context)

        exp = self.experience_store.create(
            goal=context.get("goal", ""),
            context=context.get("processed_input", ""),
            action=context.get("plan_summary", ""),
            expected_outcome="",
            actual_outcome="",
            confidence=0.5,
            reasoning_summary=context.get("reasoning_context", ""),
        )
        context["experience_id"] = exp.id

        message = self._normalize_user_message(
            context.get("processed_input") or context.get("input", "")
        )
        self._memory.append(message)
        tools = self.tool_registry.schemas() if self.tool_registry else None

        cognitive_prompt = self._build_cognitive_prompt(context)
        self._ensure_cognitive_prompt(cognitive_prompt)

        response = ""
        while True:
            self._log("[dim]AsyncMind: Calling LLM...[/]")
            result = await self.llm_client.chat_completion(
                messages=self._memory,
                tools=tools,
                temperature=1.0,
                json_response=context.get("json_response", False),
                max_completion_tokens=context.get("max_completion_tokens"),
                seed=context.get("seed"),
                reasoning_effort=context.get("reasoning_effort"),
                web_search_options=context.get("web_search_options"),
                extra_body=context.get("extra_body"),
            )
            content = result.get("content")
            tool_calls = result.get("tool_calls", [])
            finish_reason = result.get("finish_reason")

            if tool_calls:
                self._memory.append(
                    self._build_assistant_with_tool_calls(content, tool_calls)
                )
                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    self._log(f"[magenta]AsyncMind: Running tool[/] {tool_name}")
                    tool_kwargs = safe_json_loads(tool_args)
                    try:
                        tool_response = await self.tool_registry.call_async(
                            tool_name, **tool_kwargs
                        )
                    except Exception as exc:
                        tool_response = f"Tool error: {exc}"
                    if self.world_model:
                        self.world_model.ingest_tool_result(
                            tool_name, tool_args, str(tool_response)
                        )
                    self._memory.append(
                        {
                            "tool_call_id": getattr(tc, "id", ""),
                            "role": MessageRole.TOOL,
                            "name": tool_name,
                            "content": str(tool_response),
                        }
                    )
                continue

            if content:
                self._memory.append(create_message(MessageRole.ASSISTANT, content))
                response = content
                context["response"] = content
                context["last_action"] = "llm_response"
                break
            break

        context["confidence"] = self._calculate_confidence(context)
        context = self.reflector.process(context)
        graph_ids = self._add_to_graph(context)

        decision = self.decision_log.record(
            goal=context.get("goal", ""),
            options=[context.get("selected_mental_model", "direct")],
            chosen=context.get("selected_mental_model", "direct"),
            reasoning=context.get("reasoning_context", ""),
            confidence=context.get("confidence", 0.0),
        )

        context = self.learning.process(context)

        self._save_world_model()

        self.event_bus.emit(
            CognitiveEventType.EXPERIENCE_STORED,
            experience_id=exp.id,
            goal=exp.goal,
        )

        return MindOutput(
            response=response,
            confidence=context.get("confidence", 0.0),
            decisions=[{"id": decision.id, "goal": decision.goal, "chosen": decision.chosen}],
            graph_node_ids=graph_ids,
            reasoning=context.get("reasoning_context", ""),
            mental_model=context.get("selected_mental_model", ""),
            lessons=context.get("lessons", []),
            context=context,
        )

    async def run_stream(
        self,
        user_message: str | dict,
        json_response: bool = False,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        context: Dict[str, Any] = {
            "input": user_message if isinstance(user_message, str) else str(user_message),
            "goal": user_message if isinstance(user_message, str) else str(user_message),
        }

        context = self.observer.process(context)
        context = self.attention.process(context)
        context = self.thinker.process(context)

        cognitive_prompt = self._build_cognitive_prompt(context)
        self._ensure_cognitive_prompt(cognitive_prompt)

        message = self._normalize_user_message(
            context.get("processed_input") or context.get("input", "")
        )
        self._memory.append(message)
        tools = self.tool_registry.schemas() if self.tool_registry else None

        full_response = ""
        async for chunk in self.llm_client.chat_completion_stream(
            messages=self._memory,
            tools=tools,
            temperature=1.0,
            json_response=json_response,
            max_completion_tokens=max_completion_tokens,
            seed=seed,
            reasoning_effort=reasoning_effort,
            web_search_options=web_search_options,
            extra_body=extra_body,
        ):
            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                yield delta.content

        context["response"] = full_response
        context["last_action"] = "llm_stream"
        context["confidence"] = self._calculate_confidence(context)
        context = self.reflector.process(context)
        self._save_world_model()
        self._add_to_graph(context)

    def _build_cognitive_prompt(self, context: Dict[str, Any]) -> str:
        parts = []
        model_name = context.get("selected_mental_model", "")
        explanation = context.get("mental_model_explanation", "")
        reasoning = context.get("reasoning_context", "")
        plan = context.get("plan", [])
        critique = context.get("plan_critique", "")

        if model_name:
            parts.append(f"Reasoning approach: {model_name}")
        if explanation:
            parts.append(f"Framework: {explanation}")
        if reasoning:
            parts.append(f"\nReasoning:\n{reasoning}")
        if plan:
            plan_text = "\n".join(
                f"- {step.get('action', '')}" for step in plan
            )
            parts.append(f"\nPlan:\n{plan_text}")
        if critique:
            parts.append(f"\nCritique:\n{critique}")

        if self.world_model:
            wm_context = self.world_model.get_context()
            if wm_context:
                parts.append(f"\nWorld State:\n{wm_context}")

        return "\n".join(parts) if parts else ""

    def _ensure_cognitive_prompt(self, prompt: str) -> None:
        if not prompt:
            return
        base_prompt = self.prompt_registry.render(self.system_prompt_name) or ""
        combined = f"{base_prompt}\n\n{prompt}" if base_prompt else prompt
        if not self._memory or self._memory[0].get("role") not in (
            MessageRole.SYSTEM,
            MessageRole.DEVELOPER,
        ):
            self._memory.insert(0, create_message(MessageRole.DEVELOPER, combined))
        else:
            self._memory[0] = create_message(MessageRole.DEVELOPER, combined)

    def _calculate_confidence(self, context: Dict[str, Any]) -> float:
        base = 0.5
        plan_score = context.get("plan_score", 0.5)
        has_reasoning = 0.1 if context.get("reasoning_context") else 0.0
        has_response = 0.2 if context.get("response") else 0.0
        return min(1.0, base + (plan_score * 0.3) + has_reasoning + has_response)

    def _add_to_graph(self, context: Dict[str, Any]) -> List[str]:
        ids = []

        obs_node = CognitionNode(
            id="",
            type="observation",
            data={"content": context.get("input", ""), "goal": context.get("goal", "")},
        )
        self.graph.add_node(obs_node)
        ids.append(obs_node.id)

        if context.get("reasoning_context"):
            reasoning_node = CognitionNode(
                id="",
                type="reasoning",
                data={
                    "model": context.get("selected_mental_model", ""),
                    "content": context.get("reasoning_context", ""),
                },
            )
            self.graph.add_node(reasoning_node)
            ids.append(reasoning_node.id)
            self.graph.add_edge(
                CognitionEdge(
                    source_id=obs_node.id,
                    target_id=reasoning_node.id,
                    relation="led_to",
                )
            )

        response = context.get("response", "")
        if response:
            response_node = CognitionNode(
                id="",
                type="response",
                data={"content": response},
            )
            self.graph.add_node(response_node)
            ids.append(response_node.id)

            if context.get("reasoning_context"):
                self.graph.add_edge(
                    CognitionEdge(
                        source_id=ids[1] if len(ids) > 1 else obs_node.id,
                        target_id=response_node.id,
                        relation="produced",
                    )
                )

        return ids

    def _normalize_user_message(self, message: str | dict) -> dict:
        if isinstance(message, dict):
            return message
        return create_message(role=MessageRole.USER, content=message)

    def _build_assistant_with_tool_calls(self, content: Any, tool_calls: Any) -> dict:
        return {
            "role": MessageRole.ASSISTANT,
            "content": content,
            "tool_calls": [
                {
                    **({"id": tc.id} if hasattr(tc, "id") else {}),
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
