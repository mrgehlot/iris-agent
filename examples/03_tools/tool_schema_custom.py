#!/usr/bin/env python3
"""
Tool decorator with a custom schema (parameters override).
"""

from iris_agent import ToolRegistry, tool


def main() -> int:
    registry = ToolRegistry()

    @tool(
        name="search_web",
        description="Search the web for a query.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    def search_web(query: str) -> str:
        return f"Results for: {query}"

    registry.register(search_web)
    print("Schemas:", registry.schemas())
    print("search_web:", registry.call("search_web", query="iris agent"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
