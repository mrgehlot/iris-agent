"""
Example: How to add system prompts for an AI agent

This file demonstrates different ways to configure system prompts
for the iris-agent-framework.
"""

from iris_agent import (
    Agent,
    AsyncAgent,
    BaseLLMClient,
    LLMConfig,
    PromptRegistry,
    create_message,
    Role,
)


# ============================================================================
# Method 1: Simple string prompt
# ============================================================================

def example_simple_prompt():
    """Example using a simple static string prompt."""
    # Create a prompt registry and add a system prompt
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt(
        "assistant",
        "You are a helpful AI assistant that provides clear and concise answers."
    )

    # Create LLM client
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)

    # Create agent with the prompt registry
    agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="assistant"  # This matches the prompt name in registry
    )

    # Use the agent
    response = agent.run("What is Python?")
    print(response)


# ============================================================================
# Method 2: Dynamic prompt with parameters (using callable)
# ============================================================================

def example_dynamic_prompt():
    """Example using a callable prompt that accepts parameters."""
    prompt_registry = PromptRegistry()
    
    # Add a prompt as a lambda function
    prompt_registry.add_prompt(
        "customer_support",
        lambda user_name: f"You are a customer support agent for {user_name}. "
                          f"Always be polite and professional."
    )
    
    # Or use a regular function
    def get_support_prompt(user_name: str, company: str) -> str:
        return (
            f"You are a customer support agent for {company}. "
            f"You are currently helping {user_name}. "
            f"Always be polite, professional, and solution-oriented."
        )
    
    prompt_registry.add_prompt("support_agent", get_support_prompt)

    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)

    # When rendering, pass the parameters
    # Note: The prompt is rendered automatically when the agent is created
    # To use dynamic prompts, you need to render them before creating the agent
    # or update the prompt registry after creation
    
    agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="customer_support"
    )


# ============================================================================
# Method 3: Multiple prompts for different agent types
# ============================================================================

def example_multiple_prompts():
    """Example showing how to use different prompts for different agents."""
    prompt_registry = PromptRegistry()
    
    # Add multiple prompts
    prompt_registry.add_prompt(
        "assistant",
        "You are a helpful assistant that answers questions clearly."
    )
    prompt_registry.add_prompt(
        "coder",
        "You are an expert Python programmer. Provide clean, well-documented code."
    )
    prompt_registry.add_prompt(
        "writer",
        "You are a creative writing assistant. Help users write engaging content."
    )
    prompt_registry.add_prompt(
        "analyst",
        "You are a data analyst. Provide insights based on data and evidence."
    )

    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)

    # Create different agents with different prompts
    coding_agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="coder"
    )

    writing_agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="writer"
    )

    analyst_agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="analyst"
    )

    # Use each agent for its specific purpose
    code_response = coding_agent.run("Write a function to sort a list")
    writing_response = writing_agent.run("Write a short story about a robot")
    analysis_response = analyst_agent.run("Analyze this data: [1, 2, 3, 4, 5]")


# ============================================================================
# Method 4: Using AsyncAgent directly (for async contexts)
# ============================================================================

async def example_async_agent():
    """Example using AsyncAgent in an async context."""
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt(
        "assistant",
        "You are a helpful AI assistant that responds asynchronously."
    )

    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)

    async_agent = AsyncAgent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="assistant"
    )

    # Use async methods
    response = await async_agent.run("Hello!")
    print(response)

    # Or stream responses
    async for chunk in async_agent.run_stream("Tell me a story"):
        print(chunk, end="", flush=True)


# ============================================================================
# Method 5: Updating system prompt at runtime
# ============================================================================

def example_runtime_prompt_update():
    """Example showing how to update the system prompt after agent creation."""
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt("assistant", "Initial prompt")

    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)

    agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="assistant"
    )

    # Update the prompt in the registry
    prompt_registry.add_prompt(
        "assistant",
        "Updated prompt: You are now a specialized technical assistant."
    )

    # The agent's memory will be updated on the next run
    # Or you can manually update it:
    if agent.memory and agent.memory[0].get("role") == "developer":
        agent.memory[0]["content"] = prompt_registry.render("assistant")


# ============================================================================
# Method 6: Complex prompt with formatting
# ============================================================================

def example_formatted_prompt():
    """Example using string formatting in prompts."""
    prompt_registry = PromptRegistry()
    
    # Prompt with placeholders
    prompt_registry.add_prompt(
        "personalized_assistant",
        "You are {name}'s personal assistant. "
        "Your role is to help with {tasks}. "
        "Always maintain a {tone} tone."
    )

    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)

    # Render the prompt with values before creating the agent
    rendered_prompt = prompt_registry.render(
        "personalized_assistant",
        name="John",
        tasks="scheduling and email management",
        tone="professional"
    )
    
    # Add the rendered prompt as a new entry
    prompt_registry.add_prompt("assistant", rendered_prompt)

    agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="assistant"
    )


# ============================================================================
# Method 7: Using create_message to build custom messages
# ============================================================================

def example_create_message():
    """Example showing how to use create_message to build custom messages."""
    
    # Example 1: Basic text message
    user_message = create_message(
        role=Role.USER,
        content="What is machine learning?"
    )
    print("Basic message:", user_message)
    # Output: {'role': 'user', 'content': 'What is machine learning?'}
    
    # Example 2: Message with images (for vision models)
    vision_message = create_message(
        role=Role.USER,
        content="Describe what you see in these images",
        images=[
            "https://example.com/image1.jpg",
            "https://example.com/image2.png"
        ]
    )
    print("Vision message:", vision_message)
    # Output: {
    #   'role': 'user',
    #   'content': [
    #     {'type': 'text', 'text': 'Describe what you see in these images'},
    #     {'type': 'image_url', 'image_url': {'url': 'https://example.com/image1.jpg'}},
    #     {'type': 'image_url', 'image_url': {'url': 'https://example.com/image2.png'}}
    #   ]
    # }
    
    # Example 3: Message with name (for multi-user conversations)
    named_message = create_message(
        role=Role.USER,
        content="I need help with Python",
        name="John Doe"
    )
    print("Named message:", named_message)
    # Output: {'role': 'user', 'content': 'I need help with Python', 'name': 'John_Doe'}
    
    # Example 4: Image-only message (no text content)
    image_only_message = create_message(
        role=Role.USER,
        content="",  # Empty string for image-only
        images=["https://example.com/chart.png"]
    )
    print("Image-only message:", image_only_message)
    
    # Example 5: Building conversation history manually
    conversation_history = [
        create_message(Role.SYSTEM, "You are a helpful assistant."),
        create_message(Role.USER, "Hello!"),
        create_message(Role.ASSISTANT, "Hi! How can I help you?"),
        create_message(Role.USER, "What is Python?"),
    ]
    print("Conversation history:", conversation_history)
    
    # Example 6: Using create_message with agent memory
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    llm_client = BaseLLMClient(llm_config)
    
    prompt_registry = PromptRegistry()
    prompt_registry.add_prompt("assistant", "You are a helpful assistant.")
    
    agent = Agent(
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        system_prompt_name="assistant"
    )
    
    # Manually add messages to agent's memory
    agent.memory.append(create_message(Role.USER, "Tell me about AI"))
    agent.memory.append(create_message(Role.ASSISTANT, "AI stands for Artificial Intelligence..."))
    
    # Or create a message with images for vision capabilities
    vision_query = create_message(
        role=Role.USER,
        content="What's in this image?",
        images=["https://example.com/photo.jpg"]
    )
    agent.memory.append(vision_query)
    
    # Now when you run the agent, it will use the manually added messages
    # response = agent.run("Continue the conversation")


if __name__ == "__main__":
    # Uncomment to run examples:
    # example_simple_prompt()
    # example_dynamic_prompt()
    # example_multiple_prompts()
    # import asyncio; asyncio.run(example_async_agent())
    # example_runtime_prompt_update()
    # example_formatted_prompt()
    # example_create_message()
    pass
