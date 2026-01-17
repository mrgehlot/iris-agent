from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, get_args, get_origin, Mapping, Sequence


@dataclass
class ToolSpec:
    """Metadata for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    is_async: bool

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Serialize to OpenAI tool schema.

        Returns:
            A dict compatible with OpenAI tool schemas.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _schema_for_type(tp: Any) -> Dict[str, Any]:
    """
    Map a Python type annotation to a JSON schema fragment.

    Args:
        tp: Type annotation to convert.

    Returns:
        JSON schema fragment.
    """
    if tp is Any:
        return {}

    origin = get_origin(tp)
    args = get_args(tp)

    if tp is type(None):
        return {"type": "null"}

    if origin in (list, tuple, set, Sequence):
        if args:
            item_schema = _schema_for_type(args[0])
        else:
            item_schema = {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin in (dict, Mapping):
        additional = _schema_for_type(args[1]) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": additional}

    if origin is not None and str(origin).endswith("Literal"):
        values = list(args)
        if values:
            value_type = type(values[0])
            return {"type": _schema_for_type(value_type).get("type", "string"), "enum": values}
        return {"type": "string"}

    if origin is not None and str(origin).endswith("Union"):
        schemas = [_schema_for_type(arg) for arg in args]
        return {"anyOf": schemas}

    if tp in (str,):
        return {"type": "string"}
    if tp in (int,):
        return {"type": "integer"}
    if tp in (float,):
        return {"type": "number"}
    if tp in (bool,):
        return {"type": "boolean"}
    return {"type": "string"}


def _infer_parameters_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """
    Infer JSON schema from function signature annotations.

    Args:
        func: Function whose parameters should be inferred.

    Returns:
        JSON schema describing the function parameters.
    """
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = param.annotation if param.annotation is not inspect._empty else str
        properties[name] = _schema_for_type(annotation)
        if param.default is inspect._empty:
            required.append(name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to mark a function as a tool with schema metadata.

    Args:
        name: Optional tool name override.
        description: Optional tool description override.
        parameters: Optional JSON schema for parameters.

    Returns:
        Decorator that attaches a ToolSpec to the function.
    """
    def decorator(func: Callable[..., Any]):
        """
        Attach ToolSpec metadata to the wrapped function.

        Args:
            func: Function to register as a tool.

        Returns:
            The original function with ToolSpec metadata.
        """
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip() or tool_name
        tool_parameters = parameters or _infer_parameters_schema(func)
        is_async = inspect.iscoroutinefunction(func)
        spec = ToolSpec(
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters,
            func=func,
            is_async=is_async,
        )
        setattr(func, "_tool_spec", spec)
        return func

    return decorator


class ToolRegistry:
    """Register, validate, and call tools."""
    def __init__(self) -> None:
        """Initialize empty tool registry."""
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, func: Callable[..., Any]) -> ToolSpec:
        """
        Register a function as a tool and return its spec.

        Args:
            func: Callable to register.

        Returns:
            ToolSpec for the registered function.
        """
        spec = getattr(func, "_tool_spec", None)
        if spec is None:
            spec = ToolSpec(
                name=func.__name__,
                description=(func.__doc__ or "").strip() or func.__name__,
                parameters=_infer_parameters_schema(func),
                func=func,
                is_async=inspect.iscoroutinefunction(func),
            )
        self._tools[spec.name] = spec
        return spec

    def register_from(self, obj: Any) -> None:
        """
        Register all decorated tools on an object.

        Args:
            obj: Object to scan for decorated tools.
        """
        for _, member in inspect.getmembers(obj):
            if callable(member) and hasattr(member, "_tool_spec"):
                self.register(member)

    def list_tools(self) -> Dict[str, ToolSpec]:
        """
        Return a copy of the tool mapping.

        Returns:
            Dict of tool name to ToolSpec.
        """
        return dict(self._tools)

    def schemas(self) -> list[Dict[str, Any]]:
        """
        Return OpenAI-compatible schemas for all tools.

        Returns:
            List of OpenAI tool schema dicts.
        """
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def validate_args(self, name: str, args: Dict[str, Any]) -> None:
        """
        Validate tool arguments against the tool schema.

        Args:
            name: Registered tool name.
            args: Arguments to validate.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered.")
        spec = self._tools[name]
        schema = spec.parameters or {}
        props = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [r for r in required if r not in args]
        if missing:
            raise ValueError(f"Missing required args for '{name}': {', '.join(missing)}")

        for key, value in args.items():
            if key not in props:
                continue
            if not self._is_valid_value(value, props[key]):
                raise ValueError(f"Argument '{key}' does not match schema.")

    def _is_valid_value(self, value: Any, schema: Dict[str, Any]) -> bool:
        """
        Return True when value conforms to a schema snippet.

        Args:
            value: Value to validate.
            schema: JSON schema fragment.

        Returns:
            True if value conforms to schema.
        """
        if not schema:
            return True

        if "enum" in schema:
            return value in schema["enum"]

        if "anyOf" in schema:
            return any(self._is_valid_value(value, subschema) for subschema in schema["anyOf"])

        expected = schema.get("type")
        if isinstance(expected, list):
            return any(self._is_valid_value(value, {"type": t}) for t in expected)

        if expected == "null":
            return value is None
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int)
        if expected == "number":
            return isinstance(value, (int, float))
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            if not isinstance(value, dict):
                return False
            additional = schema.get("additionalProperties")
            if additional is None:
                return True
            return all(self._is_valid_value(v, additional) for v in value.values())
        if expected == "array":
            if not isinstance(value, list):
                return False
            items_schema = schema.get("items")
            if not items_schema:
                return True
            return all(self._is_valid_value(v, items_schema) for v in value)

        return True

    def call(self, name: str, **kwargs) -> Any:
        """
        Call a tool synchronously (runs async tools outside event loop).

        Args:
            name: Registered tool name.
            **kwargs: Tool arguments.

        Returns:
            Tool result.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered.")
        self.validate_args(name, kwargs)
        spec = self._tools[name]
        if spec.is_async:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(spec.func(**kwargs))
            raise RuntimeError("Tool is async. Use await ToolRegistry.call_async().")
        return spec.func(**kwargs)

    async def call_async(self, name: str, **kwargs) -> Any:
        """
        Call a tool in async contexts.

        Args:
            name: Registered tool name.
            **kwargs: Tool arguments.

        Returns:
            Tool result.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered.")
        self.validate_args(name, kwargs)
        spec = self._tools[name]
        if spec.is_async:
            return await spec.func(**kwargs)
        return spec.func(**kwargs)
