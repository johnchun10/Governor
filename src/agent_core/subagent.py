"""The ``task`` tool: delegate work to a subagent with a fresh context.

The child runs its own loop with its profile's system prompt, model and tool
subset, and only its final text comes back to the parent. Several ``task``
calls in one model response run in parallel, because the loop executes a
turn's tool calls concurrently. The child's token usage rolls up into the
parent's total.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .tools import Tool, ToolContext, ToolError
from .types import ToolSpec

if TYPE_CHECKING:
    from .agent import Runtime

DESCRIPTION = """Delegate a self-contained task to a subagent. The subagent starts with a fresh
context: it sees only your prompt, not this conversation, so include every detail it
needs (paths, goals, constraints, what to return). It returns a single final report.
Issue several task calls in one response to run subagents in parallel.

Available agents:
{agents}"""


def _run_task(ctx: ToolContext, agent: str, prompt: str) -> str:
    from .agent import Agent  # local import: avoids an import cycle with agent.py

    runtime = ctx.runtime
    if ctx.depth + 1 > runtime.max_depth:
        raise ToolError(f"subagent depth limit ({runtime.max_depth}) reached; do the work yourself")
    spec = runtime.profiles.get(agent)
    if spec is None or not spec.subagent:
        raise ToolError(f"unknown agent {agent!r}; choose one of: {', '.join(_subagents(runtime))}")
    child = Agent(spec, runtime, depth=ctx.depth + 1, parent_run_id=ctx.run_id, cwd=ctx.cwd)
    result = child.run(prompt)
    ctx.add_usage(result.usage)
    if result.stop == "error":
        raise ToolError(f"subagent {agent} failed: {result.text}")
    note = "" if result.stop == "end_turn" else f"\n\n[subagent stopped early: {result.stop}]"
    return (result.text or "(subagent returned no text)") + note


def _subagents(runtime: Runtime) -> list[str]:
    return sorted(name for name, spec in runtime.profiles.items() if spec.subagent)


def _spec(runtime: Runtime) -> ToolSpec:
    names = _subagents(runtime)
    listing = "\n".join(f"- {n}: {runtime.profiles[n].description}" for n in names)
    return ToolSpec(
        "task",
        DESCRIPTION.format(agents=listing or "(none)"),
        {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": names, "description": "which agent to run"},
                "prompt": {"type": "string", "description": "complete, self-contained instructions"},
            },
            "required": ["agent", "prompt"],
        },
    )


task_tool = Tool(
    name="task",
    description="Delegate a task to a subagent.",
    input_schema={"type": "object", "properties": {}, "required": ["agent", "prompt"]},
    fn=_run_task,
    permission="allow",  # the child's own tool calls are still permission-checked
    spec_fn=_spec,
)
