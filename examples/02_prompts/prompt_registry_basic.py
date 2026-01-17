#!/usr/bin/env python3
"""
PromptRegistry basics: static prompts and formatting.
"""

from iris_agent import PromptRegistry


def main() -> int:
    prompts = PromptRegistry()
    prompts.add_prompt("greeting", "Hello {name}!")

    print(prompts.render("greeting", name="Iris"))
    print(prompts.render("missing") or "No prompt found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
