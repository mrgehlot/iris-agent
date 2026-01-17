import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

from .async_agent import AsyncAgent
from .llm import BaseLLMClient, LLMConfig
from .prompts import PromptRegistry
from .tools import ToolRegistry




class Agent:
    """Synchronous wrapper around AsyncAgent for non-async usage."""
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_registry: Optional[PromptRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt_name: str = "assistant",
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the agent with client, prompts, and tools.

        Args:
            llm_client: LLM client instance for chat completions.
            prompt_registry: Optional prompt registry; defaults to a new one.
            tool_registry: Optional tool registry; defaults to a new one.
            system_prompt_name: Prompt key used to seed system instructions.
            enable_logging: Enable Rich logging if True.
            logger: Optional custom logger instance.
        """
        self._async_agent = AsyncAgent(
            llm_client=llm_client,
            prompt_registry=prompt_registry,
            tool_registry=tool_registry,
            system_prompt_name=system_prompt_name,
            enable_logging=enable_logging,
            logger=logger,
        )

    @property
    def memory(self):
        """
        Expose the underlying agent memory.

        Returns:
            The list of message dicts tracked by the agent.
        """
        return self._async_agent.memory

    def _run_async(self, coro):
        """
        Run a coroutine in sync context, using current loop if available.

        Args:
            coro: Awaitable to run.

        Returns:
            The coroutine result or a task if a loop is already running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return loop.create_task(coro)

    def run(self, user_message: str | dict) -> str:
        """
        Send a message and return the assistant response.

        Args:
            user_message: User input text or a pre-built message dict.

        Returns:
            The assistant response content.
        """
        result = self._run_async(self._async_agent.run(user_message))
        if asyncio.isfuture(result):
            raise RuntimeError("Agent.run called inside an event loop. Use AsyncAgent.")
        return result

    def run_stream(self, user_message: str | dict) -> AsyncGenerator[str, None]:
        """
        Streaming is not supported on the sync Agent interface.

        Args:
            user_message: User input text or a pre-built message dict.
        """
        raise RuntimeError("Use AsyncAgent.run_stream for streaming responses.")

    def call_tool(self, name: str, **kwargs) -> Any:
        """
        Call a registered tool through the underlying async agent.

        Args:
            name: Registered tool name.
            **kwargs: Tool arguments.

        Returns:
            The tool result.
        """
        result = self._run_async(self._async_agent.call_tool(name, **kwargs))
        if asyncio.isfuture(result):
            raise RuntimeError("Agent.call_tool called inside an event loop. Use AsyncAgent.")
        return result
