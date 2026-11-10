"""Tool definitions, the ``@tool`` decorator, and the registry that runs them.

A tool is a plain function whose first parameter is ``ctx: ToolContext``.
The JSON schema the model sees is generated from the remaining parameters'
type hints. Parameter descriptions come from the docstring's ``Args:``
section, and the docstring's first paragraph becomes the tool description.

    @tool
    def read_file(ctx: ToolContext, path: str, offset: int = 1) -> str:
        '''Read a text file.

        Args:
            path: file to read, relative to the working directory.
            offset: first line to return (1-based).
        '''
"""

from __future__ import annotations

import inspect
import json
import re
import time
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

from .events import Event
from .permissions import Decision
from .schema import validate
from .types import ToolResultBlock, ToolSpec, ToolUseBlock, Usage

if TYPE_CHECKING:
    from .agent import Runtime

MAX_OUTPUT_CHARS = 30_000


@dataclass
class ToolContext:
    """What a running tool can see: where it is, who called it, the shared runtime."""

    runtime: Runtime
    cwd: Path
    agent: str
    run_id: str
    depth: int
    add_usage: Callable[[Usage], None] = lambda usage: None
    """Tools that spend tokens (subagents) report them here so they roll up."""

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.cwd / p)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    permission: Decision = "ask"
    spec_fn: Callable[[Runtime], ToolSpec] | None = None
    """Build the spec at request time, for tools whose description depends on
    the runtime (e.g. ``task`` lists the available agents)."""

    def spec(self, runtime: Runtime) -> ToolSpec:
        if self.spec_fn is not None:
            return self.spec_fn(runtime)
        return ToolSpec(self.name, self.description, self.input_schema)


# --- schema generation ---------------------------------------------------------

_SIMPLE = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(tp: Any) -> dict[str, Any]:
    origin, args = get_origin(tp), get_args(tp)
    if tp in _SIMPLE:
        return {"type": _SIMPLE[tp]}
    if origin is Literal:
        return {"type": _SIMPLE[type(args[0])], "enum": list(args)}
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        return _json_type(non_none[0]) if len(non_none) == 1 else {}
    if origin is list or tp is list:
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin is dict or tp is dict:
        return {"type": "object"}
    return {}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    doc = inspect.cleandoc(doc or "")
    summary, _, rest = doc.partition("\nArgs:")
    params: dict[str, str] = {}
    current = None
    for line in rest.splitlines():
        m = re.match(r"^\s{2,}(\w+)\s*(?:\([^)]*\))?:\s*(.*)$", line)
        if m:
            current = m.group(1)
            params[current] = m.group(2).strip()
        elif current and line.strip():
            params[current] += " " + line.strip()
        elif not line.strip():
            current = None
    return summary.strip(), params


def schema_from_function(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    hints = typing.get_type_hints(fn)
    summary, param_docs = _parse_docstring(fn.__doc__ or "")
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in list(inspect.signature(fn).parameters.items())[1:]:  # skip ctx
        prop = _json_type(hints.get(name, str))
        if name in param_docs:
            prop["description"] = param_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        props[name] = prop
    # The function can't take arguments it doesn't declare, so the schema says so.
    return summary, {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    permission: Decision = "ask",
) -> Any:
    """Turn a function into a Tool. ``permission`` is its default policy."""

    def wrap(f: Callable[..., Any]) -> Tool:
        description, schema = schema_from_function(f)
        return Tool(name or f.__name__, description, schema, f, permission)

    return wrap(fn) if fn is not None else wrap


# --- registry ------------------------------------------------------------------


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    dropped = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n\n... [{dropped} characters truncated] ...\n\n{text[-half:]}"


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, *tools: Tool) -> ToolRegistry:
        for t in tools:
            self.tools[t.name] = t
        return self

    def names(self, patterns: list[str]) -> list[str]:
        """Expand a profile's tool list: ``*`` means every tool, ``-name`` removes one."""
        selected: list[str] = []
        for p in patterns:
            if p == "*":
                selected += [n for n in self.tools if n not in selected]
            elif p.startswith("-"):
                selected = [n for n in selected if n != p[1:]]
            elif p in self.tools and p not in selected:
                selected.append(p)
        return selected

    def execute(
        self, call: ToolUseBlock, ctx: ToolContext, allowed: list[str], profile_permissions: dict[str, Decision]
    ) -> ToolResultBlock:
        return self.run(call, ctx, allowed, profile_permissions)[0]

    def run(
        self, call: ToolUseBlock, ctx: ToolContext, allowed: list[str], profile_permissions: dict[str, Decision]
    ) -> tuple[ToolResultBlock, Outcome]:
        """Run one tool call and say how it went.

        ``invalid`` means the model's call was wrong (unknown tool, or arguments
        that fail the tool's schema) and the result says exactly what to fix;
        ``denied`` means permissions refused it; ``error`` means the tool ran and
        failed. The loop counts ``invalid`` calls per tool (``AgentSpec.max_invalid_calls``).
        """
        runtime = ctx.runtime
        emit = runtime.events.emit
        emit(
            Event(
                "tool_call",
                ctx.run_id,
                ctx.agent,
                ctx.depth,
                data={"id": call.id, "name": call.name, "input": call.input},
            )
        )
        started = time.monotonic()
        outcome: Outcome = "invalid"
        t = self.tools.get(call.name)
        if t is None or call.name not in allowed:
            content = f"error: unknown tool {call.name!r}. Available tools: {', '.join(allowed)}"
        elif problems := validate(call.input, t.spec(runtime).input_schema):
            content = f"error: invalid arguments for {call.name}: " + "; ".join(problems)
        elif refusal := runtime.permissions.check(call.name, call.input, t.permission, profile_permissions):
            content, outcome = refusal, "denied"
        else:
            try:
                result = t.fn(ctx, **call.input)
                content = result if isinstance(result, str) else json.dumps(result, default=str)
                outcome = "ok"
            except ToolError as exc:
                content, outcome = f"error: {exc}", "error"
            except Exception as exc:  # a broken tool must never crash the loop
                content, outcome = f"error: {type(exc).__name__}: {exc}", "error"
        content = _truncate(content)
        emit(
            Event(
                "tool_result",
                ctx.run_id,
                ctx.agent,
                ctx.depth,
                data={
                    "id": call.id,
                    "name": call.name,
                    "content": content,
                    "is_error": outcome != "ok",
                    "outcome": outcome,
                    "duration_s": round(time.monotonic() - started, 3),
                },
            )
        )
        return ToolResultBlock(call.id, content, outcome != "ok"), outcome


Outcome = Literal["ok", "invalid", "denied", "error"]


class ToolError(Exception):
    """Raise from a tool to return a clean error message to the model."""
