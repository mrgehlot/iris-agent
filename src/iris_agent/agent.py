import asyncio
import logging
from typing import Any, Generator, Optional

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

    def run(
        self,
        user_message: str | dict,
        json_response: bool = False,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
        extra_body: dict | None = None,
    ) -> str:
        """
        Send a message and return the assistant response.

        Args:
            user_message: User input text or a pre-built message dict.
            json_response: Request JSON-only response when supported.
            max_completion_tokens: Cap the completion token count.
            seed: Optional seed for deterministic sampling.
            reasoning_effort: Optional reasoning effort hint for supported models.
            web_search_options: Optional web search options for supported models.
            extra_body: Optional provider-specific request body overrides.

        Returns:
            The assistant response content.
        """
        result = self._run_async(
            self._async_agent.run(
                user_message,
                json_response=json_response,
                max_completion_tokens=max_completion_tokens,
                seed=seed,
                reasoning_effort=reasoning_effort,
                web_search_options=web_search_options,
                extra_body=extra_body,
            )
        )
        if asyncio.isfuture(result):
            raise RuntimeError("Agent.run called inside an event loop. Use AsyncAgent.")
        return result

    def run_stream(
        self,
        user_message: str | dict,
        json_response: bool = False,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        web_search_options: dict | None = None,
        extra_body: dict | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream responses from the async agent in a sync-friendly way.

        Args:
            user_message: User input text or a pre-built message dict.
            json_response: Request JSON-only response when supported.
            max_tokens: Cap the streamed completion tokens.
            seed: Optional seed for deterministic sampling.
            reasoning_effort: Optional reasoning effort hint for supported models.
            web_search_options: Optional web search options for supported models.
            extra_body: Optional provider-specific request body overrides.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Agent.run_stream called inside an event loop. Use AsyncAgent.")

        async_gen = self._async_agent.run_stream(
            user_message,
            json_response=json_response,
            max_tokens=max_tokens,
            seed=seed,
            reasoning_effort=reasoning_effort,
            web_search_options=web_search_options,
            extra_body=extra_body,
        )

        def _iterator() -> Generator[str, None, None]:
            loop = asyncio.new_event_loop()
            try:
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                    except StopAsyncIteration:
                        break
                    else:
                        yield chunk
            finally:
                loop.run_until_complete(async_gen.aclose())
                loop.close()

        return _iterator()

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
