#!/usr/bin/env python3
"""
Verification script to check if the package is set up correctly.
Run this after installing the package to verify everything works.
"""

import sys
import importlib


def check_imports():
    """Check if all main modules can be imported."""
    print("Checking imports...")
    
    try:
        from iris_agent import (
            Agent,
            AsyncAgent,
            BaseLLMClient,
            LLMConfig,
            LLMProvider,
            PromptRegistry,
            ToolRegistry,
            create_message,
            Role,
            tool,
        )
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_basic_functionality():
    """Check basic functionality without API calls."""
    print("\nChecking basic functionality...")
    
    try:
        from iris_agent import PromptRegistry, ToolRegistry, create_message, Role, tool
        
        # Test PromptRegistry
        prompts = PromptRegistry()
        prompts.add_prompt("test", "Test prompt")
        assert prompts.render("test") == "Test prompt"
        print("✅ PromptRegistry works")
        
        # Test create_message
        msg = create_message(Role.USER, "Hello")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        print("✅ create_message works")
        
        # Test ToolRegistry
        tools = ToolRegistry()
        
        @tool(description="Test tool")
        def test_tool(x: int) -> int:
            return x * 2
        
        tools.register(test_tool)
        result = tools.call("test_tool", x=5)
        assert result == 10
        print("✅ ToolRegistry works")
        
        return True
    except Exception as e:
        print(f"❌ Functionality check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_package_info():
    """Check package metadata."""
    print("\nChecking package info...")
    
    try:
        import iris_agent
        print(f"✅ Package location: {iris_agent.__file__}")
        
        # Check __all__
        if hasattr(iris_agent, "__all__"):
            print(f"✅ Exports: {', '.join(iris_agent.__all__)}")
        
        return True
    except Exception as e:
        print(f"❌ Package info check failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 50)
    print("Iris Agent Package Verification")
    print("=" * 50)
    
    checks = [
        check_imports,
        check_basic_functionality,
        check_package_info,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All checks passed! Package is ready to use.")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
