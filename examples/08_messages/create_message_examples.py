#!/usr/bin/env python3
"""
Message creation examples using create_message and Role.
"""

from iris_agent import Role, create_message


def main() -> int:
    text_msg = create_message(Role.USER, "Hello")
    print("Text message:", text_msg)

    image_msg = create_message(
        Role.USER,
        "Describe this image",
        images=["https://example.com/image.jpg"],
    )
    print("Image message:", image_msg)

    named_msg = create_message(Role.USER, "Hello", name="John Doe")
    print("Named message:", named_msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
