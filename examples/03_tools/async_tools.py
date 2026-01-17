#!/usr/bin/env python3
"""
Async tools with ToolRegistry.call_async.
"""

import asyncio

from iris_agent import ToolRegistry, tool


async def main() -> int:
    registry = ToolRegistry()

    @tool(description="Async add")
    async def add_async(a: int, b: int) -> int:
        return a + b

    registry.register(add_async)
    result = await registry.call_async("add_async", a=5, b=7)
    print("add_async(5, 7) =", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
