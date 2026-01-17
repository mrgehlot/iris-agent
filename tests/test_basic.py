"""Basic unit tests for iris-agent-framework."""

import pytest
from iris_agent import (
    PromptRegistry,
    ToolRegistry,
    create_message,
    Role,
    tool,
)


def test_create_message():
    """Test create_message function."""
    # Basic message
    msg = create_message(Role.USER, "Hello")
    assert msg["role"] == "user"
    assert msg["content"] == "Hello"
    
    # Message with name
    msg_with_name = create_message(Role.USER, "Hello", name="John Doe")
    assert msg_with_name["name"] == "John_Doe"
    
    # Message with images
    msg_with_images = create_message(
        Role.USER,
        "Describe this",
        images=["https://example.com/image.jpg"]
    )
    assert isinstance(msg_with_images["content"], list)
    assert len(msg_with_images["content"]) == 2  # text + image
    assert msg_with_images["content"][0]["type"] == "text"
    assert msg_with_images["content"][1]["type"] == "image_url"
    
    # Image-only message
    msg_image_only = create_message(
        Role.USER,
        "",
        images=["https://example.com/image.jpg"]
    )
    assert isinstance(msg_image_only["content"], list)
    assert len(msg_image_only["content"]) == 1  # only image


def test_prompt_registry():
    """Test PromptRegistry."""
    registry = PromptRegistry()
    
    # Add simple prompt
    registry.add_prompt("test", "You are a test assistant.")
    rendered = registry.render("test")
    assert rendered == "You are a test assistant."
    
    # Add callable prompt
    def dynamic_prompt(name: str) -> str:
        return f"You are {name}'s assistant."
    
    registry.add_prompt("dynamic", dynamic_prompt)
    rendered = registry.render("dynamic", name="John")
    assert rendered == "You are John's assistant."
    
    # Test non-existent prompt
    assert registry.render("nonexistent") is None
    
    # Test get_prompt
    prompt = registry.get_prompt("test")
    assert prompt == "You are a test assistant."


def test_tool_registry():
    """Test ToolRegistry and tool decorator."""
    registry = ToolRegistry()
    
    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b
    
    registry.register(add)
    
    # Check tool is registered
    assert "add" in registry._tools
    spec = registry._tools["add"]
    assert spec.name == "add"
    assert spec.description == "Add two numbers"
    assert spec.is_async is False
    
    # Get schemas
    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "add"
    assert "parameters" in schemas[0]["function"]
    
    # Call tool synchronously
    result = registry.call("add", a=2, b=3)
    assert result == 5


def test_tool_validation():
    """Test tool argument validation."""
    registry = ToolRegistry()
    
    @tool(description="Test validation")
    def validate_test(value: str) -> str:
        return value
    
    registry.register(validate_test)
    
    # Valid call
    result = registry.call("validate_test", value="test")
    assert result == "test"
    
    # Invalid call - should raise error
    with pytest.raises((KeyError, ValueError, TypeError)):
        registry.call("validate_test", wrong_param="test")


def test_tool_with_optional_params():
    """Test tool with optional parameters."""
    registry = ToolRegistry()
    
    @tool(description="Test optional")
    def test_optional(required: str, optional: str = "default") -> str:
        return f"{required}:{optional}"
    
    registry.register(test_optional)
    
    # Call with both params
    result = registry.call("test_optional", required="req", optional="opt")
    assert result == "req:opt"
    
    # Call with only required
    result = registry.call("test_optional", required="req")
    assert result == "req:default"


def test_async_tool():
    """Test async tool registration."""
    registry = ToolRegistry()
    
    @tool(description="Async test")
    async def async_add(a: int, b: int) -> int:
        return a + b
    
    registry.register(async_add)
    
    spec = registry._tools["async_add"]
    assert spec.is_async is True


def test_tool_schema_inference():
    """Test that tool schema is correctly inferred."""
    registry = ToolRegistry()
    
    @tool(description="Test schema")
    def schema_test(
        text: str,
        number: int,
        decimal: float,
        flag: bool,
        items: list[str],
    ) -> str:
        return "test"
    
    registry.register(schema_test)
    
    schemas = registry.schemas()
    schema = schemas[0]["function"]["parameters"]
    
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["number"]["type"] == "integer"
    assert schema["properties"]["decimal"]["type"] == "number"
    assert schema["properties"]["flag"]["type"] == "boolean"
    assert schema["properties"]["items"]["type"] == "array"


def test_role_constants():
    """Test Role constants."""
    from iris_agent import Role
    
    assert Role.SYSTEM == "system"
    assert Role.DEVELOPER == "developer"
    assert Role.USER == "user"
    assert Role.ASSISTANT == "assistant"
    assert Role.TOOL == "tool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
