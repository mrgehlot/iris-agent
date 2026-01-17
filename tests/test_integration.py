"""Integration tests (require API keys)."""

import pytest
import os
from iris_agent import (
    Agent,
    AsyncAgent,
    BaseLLMClient,
    LLMConfig,
    LLMProvider,
    PromptRegistry,
    ToolRegistry,
    tool,
)


@pytest.mark.integration
def test_agent_basic():
    """Test basic agent functionality (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    # Setup
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful assistant. Be concise.")
    
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=api_key
    )
    llm_client = BaseLLMClient(llm_config)
    
    agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompts,
        system_prompt_name="assistant",
        enable_logging=False,
    )
    
    # Test that memory is initialized
    assert len(agent.memory) > 0
    assert agent.memory[0]["role"] == "developer"
    
    # Test basic run
    response = agent.run("Say hello in one word.")
    assert response is not None
    assert len(response) > 0
    print(f"Agent response: {response}")


@pytest.mark.integration
def test_agent_with_tools():
    """Test agent with tools (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    # Setup
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are a helpful math assistant.")
    
    tools = ToolRegistry()
    
    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b
    
    @tool(description="Multiply two numbers")
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b
    
    tools.register(add)
    tools.register(multiply)
    
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=api_key
    )
    llm_client = BaseLLMClient(llm_config)
    
    agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompts,
        tool_registry=tools,
        enable_logging=False,
    )
    
    # Test with tool usage
    response = agent.run("What is 5 + 3? Use the add tool.")
    assert response is not None
    assert len(response) > 0
    print(f"Agent response: {response}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_agent():
    """Test async agent (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=api_key
    )
    llm_client = BaseLLMClient(llm_config)
    
    agent = AsyncAgent(
        llm_client=llm_client,
        prompt_registry=prompts,
        system_prompt_name="assistant"
    )
    
    # Test memory initialization
    assert len(agent.memory) > 0
    
    # Test async run
    response = await agent.run("Say hello.")
    assert response is not None
    assert len(response) > 0
    print(f"Async agent response: {response}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_agent_streaming():
    """Test async agent streaming (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    prompts = PromptRegistry()
    prompts.add_prompt("assistant", "You are helpful.")
    
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=api_key
    )
    llm_client = BaseLLMClient(llm_config)
    
    agent = AsyncAgent(
        llm_client=llm_client,
        prompt_registry=prompts,
        system_prompt_name="assistant"
    )
    
    chunks = []
    async for chunk in agent.run_stream("Say hello in 3 words"):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0
    print(f"Streamed response: {full_response}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
