import asyncio
import logging
from typing import Any, AsyncGenerator, List, Optional

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover - optional dependency
    RichHandler = None

from .llm import BaseLLMClient
from .messages import create_message
from .prompts import PromptRegistry
from .tools import ToolRegistry
from .types import Role


class AsyncAgent:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_registry: Optional[PromptRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt_name: str = "assistant",
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.tool_registry = tool_registry or ToolRegistry()
        self.system_prompt_name = system_prompt_name
        self.memory: List[dict] = []
        self.logger = logger or (self._setup_logger() if enable_logging else None)
        self._ensure_system_prompt()

    @staticmethod
    def create_message(role: str, content: str) -> dict:
        return create_message(role=role, content=content)

    def _ensure_system_prompt(self) -> None:
        prompt = self.prompt_registry.render(self.system_prompt_name)
        if not prompt:
            return
        if not self.memory or self.memory[0].get("role") not in (Role.SYSTEM, Role.DEVELOPER):
            self.memory.insert(0, self.create_message(Role.DEVELOPER, prompt))
        else:
            self.memory[0] = self.create_message(Role.DEVELOPER, prompt)

    def _setup_logger(self) -> logging.Logger:
        if RichHandler is None:
            raise ImportError(
                "Install rich optional dependency: pip install iris-agent-framework[rich]"
            )
        logger = logging.getLogger("iris_agent")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = RichHandler(rich_tracebacks=True, markup=True, show_time=False)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)

    @staticmethod
    def _truncate(value: Any, limit: int = 200) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}…"

    async def run(
        self,
        user_message: str,
        temperature: float = 1.0,
        json_response: bool = False,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
    ) -> str:
        self.memory.append(self.create_message(Role.USER, user_message))
        tools = self.tool_registry.schemas() if self.tool_registry else None
        self._log(f"[bold cyan]User[/]: {self._truncate(user_message)}")

        while True:
            self._log("[dim]Calling LLM...[/]")
            result = await self.llm_client.chat_completion(
                messages=self.memory,
                tools=tools,
                temperature=temperature,
                json_response=json_response,
                max_completion_tokens=max_completion_tokens,
                seed=seed,
                reasoning_effort=reasoning_effort,
                web_search_options=web_search_options,
            )
            content = result.get("content")
            tool_calls = result.get("tool_calls", [])
            finish_reason = result.get("finish_reason")
            if tool_calls:
                self._log(f"[yellow]Tool calls requested[/]: {len(tool_calls)}")
            if finish_reason:
                self._log(f"[dim]Finish reason[/]: {finish_reason}")

            if tool_calls:
                self.memory.append(
                    {
                        "role": Role.ASSISTANT,
                        "content": content,
                        "tool_calls": [
                            {
                                **({"id": tc.id} if hasattr(tc, "id") else {}),
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                                **(
                                    {"extra_content": tc.extra_content}
                                    if hasattr(tc, "extra_content") and tc.extra_content is not None
                                    else {}
                                ),
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    self._log(
                        f"[magenta]Running tool[/] {tool_name} "
                        f"args={self._truncate(tool_args)}"
                    )
                    tool_kwargs = self._safe_json_loads(tool_args)
                    try:
                        tool_response = await self.tool_registry.call(tool_name, **tool_kwargs)
                    except Exception as exc:
                        tool_response = f"Tool error: {exc}"
                    self._log(
                        f"[green]Tool response[/] {tool_name}: "
                        f"{self._truncate(tool_response)}"
                    )
                    self.memory.append(
                        {
                            **({"tool_call_id": tc.id} if hasattr(tc, "id") else {}),
                            "role": Role.TOOL,
                            "name": tool_name,
                            "content": str(tool_response),
                        }
                    )
                continue

            if finish_reason == "stop" and content:
                self._log(f"[bold green]Assistant[/]: {self._truncate(content)}")
                self.memory.append(self.create_message(Role.ASSISTANT, content))
                return content

            if content:
                self._log(f"[bold green]Assistant[/]: {self._truncate(content)}")
                self.memory.append(self.create_message(Role.ASSISTANT, content))
                return content
            return ""

    async def run_stream(
        self,
        user_message: str,
        temperature: float = 1.0,
        json_response: bool = False,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        self.memory.append(self.create_message(Role.USER, user_message))
        tools = self.tool_registry.schemas() if self.tool_registry else None
        self._log(f"[bold cyan]User[/]: {self._truncate(user_message)}")

        while True:
            tool_call_chunks: dict[int, dict] = {}
            full_response = ""

            self._log("[dim]Streaming LLM response...[/]")
            async for chunk in self.llm_client.chat_completion_stream(
                messages=self.memory,
                tools=tools,
                temperature=temperature,
                json_response=json_response,
                max_tokens=max_tokens,
                seed=seed,
                reasoning_effort=reasoning_effort,
                web_search_options=web_search_options,
            ):
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    yield delta.content
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        index = tc_chunk.index
                        if index not in tool_call_chunks:
                            tool_call_chunks[index] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_chunk.id:
                            tool_call_chunks[index]["id"] = tc_chunk.id
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                tool_call_chunks[index]["function"]["name"] = tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                tool_call_chunks[index]["function"]["arguments"] += (
                                    tc_chunk.function.arguments
                                )
                        if tc_chunk.extra_content:
                            tool_call_chunks[index]["extra_content"] = tc_chunk.extra_content

            tool_calls = [tool_call_chunks[i] for i in sorted(tool_call_chunks.keys())]
            if tool_calls:
                self._log(f"[yellow]Tool calls requested[/]: {len(tool_calls)}")

            assistant_message = {"role": Role.ASSISTANT, "content": full_response}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            self.memory.append(assistant_message)

            if not tool_calls:
                if full_response:
                    self._log(f"[bold green]Assistant[/]: {self._truncate(full_response)}")
                break

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                self._log(
                    f"[magenta]Running tool[/] {tool_name} "
                    f"args={self._truncate(tool_args)}"
                )
                tool_kwargs = self._safe_json_loads(tool_args)
                try:
                    tool_response = await self.tool_registry.call(tool_name, **tool_kwargs)
                except Exception as exc:
                    tool_response = f"Tool error: {exc}"
                self._log(
                    f"[green]Tool response[/] {tool_name}: "
                    f"{self._truncate(tool_response)}"
                )
                self.memory.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": Role.TOOL,
                        "name": tool_name,
                        "content": str(tool_response),
                    }
                )

    async def call_tool(self, name: str, **kwargs) -> Any:
        return await self.tool_registry.call(name, **kwargs)

    @staticmethod
    def _safe_json_loads(value: str) -> dict:
        try:
            import json

            return json.loads(value) if value else {}
        except Exception:
            return {}
