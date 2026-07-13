import json
import logging
from typing import Any

from rich.logging import RichHandler


def safe_json_loads(value: str) -> dict:
    try:
        return json.loads(value) if value else {}
    except Exception:
        return {}


def truncate(value: Any, limit: int = 200) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def setup_agent_logger() -> logging.Logger:
    logger = logging.getLogger("iris_agent")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True, show_time=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
