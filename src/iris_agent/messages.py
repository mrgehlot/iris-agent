from __future__ import annotations

from typing import List, Literal

RoleLiteral = Literal["system", "developer", "user", "assistant", "tool"]


def create_message(
    role: RoleLiteral,
    content: str | None = None,
    name: str | None = None,
    images: List[str] | None = None,
) -> dict:
    """
    Create a message object compatible with chat completions.
    Mirrors the Iris.create_message behavior from iris/flexibles/async_agent.py.

    Args:
        role: The chat role ("system", "developer", "user", "assistant", "tool").
        content: Text content for the message. Use "" for image-only messages.
        name: Optional name for multi-user contexts; spaces are converted to underscores.
        images: Optional list of image URLs for multimodal messages.

    Returns:
        A message dict ready to be sent to the LLM client.
    """
    message: dict = {"role": role}
    if name:
        message["name"] = name.replace(" ", "_")
    if not images:
        message["content"] = content
    else:
        message["content"] = []
        if content != "":
            message["content"].append({"type": "text", "text": content})
        for image in images:
            message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                }
            )
    return message
