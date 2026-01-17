#!/usr/bin/env python3
"""
PromptRegistry with dynamic prompt functions.
"""

from iris_agent import PromptRegistry


def main() -> int:
    prompts = PromptRegistry()

    def assistant_for(name: str) -> str:
        return f"You are {name}'s assistant. Be concise."

    prompts.add_prompt("assistant", assistant_for)
    print(prompts.render("assistant", name="Abhishek"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
