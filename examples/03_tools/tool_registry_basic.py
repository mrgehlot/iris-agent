#!/usr/bin/env python3
"""
ToolRegistry basics: register tools and call them directly.
"""

from iris_agent import ToolRegistry, tool


def main() -> int:
    registry = ToolRegistry()

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    registry.register(add)
    print("Schemas:", registry.schemas())
    print("add(2, 3) =", registry.call("add", a=2, b=3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
