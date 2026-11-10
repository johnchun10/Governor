"""The agent loop.

State is a list of messages, and each model call moves it forward:

    call model -> append its reply -> if it asked for tools, run them (in
    parallel) and append the results -> repeat until it stops asking.

``Runtime`` is the environment shared by a whole tree of agents: tool
registry, profiles, model resolution, provider lookup, event bus,
permissions, context providers and the stop event. All of them are injected;
core holds the loop and the contracts, Governor decides the concrete pieces.

``Agent`` is one run of one profile within that runtime. Subagents are new
``Agent`` objects created by the ``task`` tool with ``depth + 1``.

Three policies live in the loop and nowhere else:

* a reply cut off at ``max_tokens`` with tool calls in it has those calls
  refused and the model asked to retry with smaller inputs;
* a tool the model keeps calling wrongly (``max_invalid_calls`` times in a
  row) ends the run with ``invalid_tool_calls`` instead of looping;
* with ``output_schema`` set, the final text must be JSON matching it; one
  corrective turn is allowed, then the run ends with ``invalid_output``.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .context import ContextProvider, RunView
from .events import Event, EventBus
from .models import ModelInfo
from .permissions import Decision, PermissionPolicy, StaticPolicy
from .provider import Provider, ProviderCancelled, ProviderError
from .schema import parse_json, validate
from .tools import Outcome, ToolContext, ToolRegistry
from .types import Delta, Message, ModelRequest, StopReason, TextBlock, ToolResultBlock, ToolUseBlock, Usage


@dataclass
class AgentSpec:
    """What an agent is: prompt, model, tools, limits. Usually loaded from a profile."""

    name: str
    description: str = ""
    system_prompt: str = ""
    model: str | None = None
    tools: list[str] = field(default_factory=lambda: ["*"])
    max_turns: int = 50
    max_tokens: int = 16000
    subagent: bool = True
    permissions: dict[str, Decision] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    """JSON Schema the final reply must match; ``RunResult.structured`` holds the parsed value."""
    max_invalid_calls: int = 3
    """Consecutive invalid calls to one tool (unknown tool, bad arguments) that end the run."""


@dataclass
class Runtime:
    """Everything a tree of agents shares. Governor builds one (``governor.build_runtime``);
    core never decides which models, providers, tools or profiles exist."""

    registry: ToolRegistry
    profiles: dict[str, AgentSpec]
    resolve_model: Callable[[str | None], ModelInfo]
    """Turn a profile's model name (alias or id, or None for the default) into a ModelInfo."""
    provider_for: Callable[[ModelInfo], Provider]
    events: EventBus = field(default_factory=EventBus)
    permissions: PermissionPolicy = field(default_factory=StaticPolicy)
    context_providers: list[ContextProvider] = field(default_factory=list)
    max_depth: int = 2
    max_parallel_tools: int = 8
    model_override: str | None = None
    """Force every agent in the tree onto this model (e.g. ``--model-all`` for A/B runs)."""
    stop: threading.Event | None = None
    """Set it to stop every run in the tree: the model call in progress is cancelled,
    pending tool calls are refused, and each run ends with ``cancelled``."""


@dataclass
class RunResult:
    text: str
    stop: str
    """``end_turn``, ``max_turns``, ``max_tokens``, ``content_filtered``, ``cancelled``,
    ``invalid_tool_calls``, ``invalid_output`` or ``error``."""
    messages: list[Message]
    usage: Usage
    run_id: str
    turns: int
    structured: Any = None
    """The parsed final reply when the spec has an ``output_schema`` and the reply matched it."""


_MAX_TOKENS_NOTE = "Your previous response was cut off by the output token limit. Retry with smaller tool inputs."


class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        runtime: Runtime,
        *,
        depth: int = 0,
        parent_run_id: str | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.spec = spec
        self.runtime = runtime
        self.depth = depth
        self.parent_run_id = parent_run_id
        self.cwd = (cwd or Path.cwd()).resolve()
        self.model = runtime.resolve_model(runtime.model_override or spec.model)
        self.provider = runtime.provider_for(self.model)

    # -- helpers ------------------------------------------------------------------

    def _emit(self, run_id: str, type_: str, **data: Any) -> None:
        self.runtime.events.emit(Event(type_, run_id, self.spec.name, self.depth, self.parent_run_id, data))

    def _stopped(self) -> bool:
        return self.runtime.stop is not None and self.runtime.stop.is_set()

    def _tool_names(self) -> list[str]:
        names = self.runtime.registry.names(self.spec.tools)
        if self.depth >= self.runtime.max_depth:
            names = [n for n in names if n != "task"]
        return names

    def _system_prompt(self, view: RunView) -> str:
        parts = [self.spec.system_prompt]
        parts += [s for p in self.runtime.context_providers if (s := p.system(view))]
        return "\n\n".join(p for p in parts if p)

    def _inject_context(self, messages: list[Message], view: RunView, run_id: str) -> None:
        extra = [s for p in self.runtime.context_providers if (s := p.before_turn(view))]
        if not extra:
            return
        text = "<context>\n" + "\n\n".join(extra) + "\n</context>"
        last = messages[-1]  # always the not-yet-sent user message
        messages[-1] = replace(last, content=(*last.content, TextBlock(text)))
        self._emit(run_id, "context", turn=view.turn, text=text)

    def _check_output(self, text: str) -> tuple[Any, str | None]:
        """Parse and validate a final reply against the spec's output schema. Returns (value, problem)."""
        schema = self.spec.output_schema or {}
        value, error = parse_json(text)
        if error:
            return None, error
        problems = validate(value, schema)
        return (None, "; ".join(problems)) if problems else (value, None)

    # -- the loop -------------------------------------------------------------------

    def run(self, task: str, history: list[Message] | None = None) -> RunResult:
        """Run until the model stops calling tools. ``history`` continues a chat."""
        run_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        messages: list[Message] = [*(history or []), Message.user(task)]
        usage = Usage()
        usage_lock = threading.Lock()

        def add_usage(u: Usage) -> None:
            nonlocal usage
            with usage_lock:
                usage = usage + u

        tool_names = self._tool_names()
        tool_specs = tuple(self.runtime.registry.tools[n].spec(self.runtime) for n in tool_names)
        view = RunView(self.spec, task, self.cwd, self.depth, 0, tuple(messages))
        system = self._system_prompt(view)
        ctx = ToolContext(self.runtime, self.cwd, self.spec.name, run_id, self.depth, add_usage)
        self._emit(
            run_id, "run_start", task=task, model=self.model.model_id, system=system, tools=tool_names, depth=self.depth
        )

        stop, text, turn, structured = "max_turns", "", 0, None
        invalid_streak: dict[str, int] = {}
        output_retried = False
        for turn in range(1, self.spec.max_turns + 1):
            if self._stopped():
                stop = "cancelled"
                break
            self._inject_context(messages, replace(view, turn=turn, messages=tuple(messages)), run_id)
            request = ModelRequest(
                model=self.model.model_id,
                messages=tuple(messages),
                system=system,
                tools=tool_specs,
                max_tokens=self.spec.max_tokens,
                cache=self.model.supports_cache,
                params=self.spec.params,
                output_schema=self.spec.output_schema,
            )

            def on_delta(d: Delta, turn: int = turn) -> None:
                self._emit(run_id, "delta", turn=turn, kind=d.kind, text=d.text, index=d.index)

            try:
                resp = self.provider.complete(request, on_delta=on_delta, stop=self.runtime.stop)
            except ProviderCancelled as exc:
                stop, text = "cancelled", exc.partial.text if exc.partial else ""
                break
            except ProviderError as exc:
                stop, text = "error", str(exc)
                self._emit(run_id, "error", message=text)
                break
            call_usage = replace(resp.usage, cost_usd=self.model.cost(resp.usage))
            add_usage(call_usage)
            messages.append(resp.message)
            self._emit(
                run_id,
                "model_call",
                turn=turn,
                message=resp.message.to_dict(),
                stop_reason=resp.stop_reason.value,
                usage=call_usage.to_dict(),
                latency_s=round(resp.latency_s, 3),
            )
            text = resp.message.text
            tool_uses = resp.message.tool_uses

            if resp.stop_reason is StopReason.MAX_TOKENS:
                if not tool_uses:
                    stop = "max_tokens"
                    break
                # Tool inputs may be truncated: refuse them and ask the model to retry.
                results: list[Any] = [
                    ToolResultBlock(t.id, "error: not run, your response hit max_tokens", True) for t in tool_uses
                ]
                messages.append(Message("user", (*results, TextBlock(_MAX_TOKENS_NOTE))))
                continue
            if not tool_uses:
                if resp.stop_reason is StopReason.CONTENT_FILTERED:
                    stop = "content_filtered"
                    break
                if self.spec.output_schema is not None:
                    structured, problem = self._check_output(text)
                    if problem and not output_retried:
                        output_retried = True
                        messages.append(
                            Message.user(
                                "Your reply must be a single JSON value matching the required schema. "
                                f"Problem: {problem}. Reply again with only the corrected JSON."
                            )
                        )
                        continue
                    if problem:
                        stop = "invalid_output"
                        break
                stop = "end_turn"
                break

            results, outcomes = self._run_tools(tool_uses, ctx, tool_names)
            messages.append(Message("user", tuple(results)))
            for call, outcome in zip(tool_uses, outcomes, strict=True):
                invalid_streak[call.name] = invalid_streak.get(call.name, 0) + 1 if outcome == "invalid" else 0
            if any(n >= self.spec.max_invalid_calls for n in invalid_streak.values()):
                stop = "invalid_tool_calls"
                break

        result = RunResult(text, stop, messages, usage, run_id, turn, structured)
        self._emit(
            run_id,
            "run_end",
            stop=stop,
            text=text,
            usage=usage.to_dict(),
            turns=turn,
            duration_s=round(time.monotonic() - started, 3),
        )
        return result

    def _run_tools(
        self, tool_uses: list[ToolUseBlock], ctx: ToolContext, allowed: list[str]
    ) -> tuple[list[ToolResultBlock], list[Outcome]]:
        registry, perms = self.runtime.registry, self.spec.permissions

        def one(tu: ToolUseBlock) -> tuple[ToolResultBlock, Outcome]:
            if self._stopped():
                return ToolResultBlock(tu.id, "error: not run, stopped by user", True), "denied"
            return registry.run(tu, ctx, allowed, perms)

        if len(tool_uses) == 1:
            pairs = [one(tool_uses[0])]
        else:
            workers = min(len(tool_uses), self.runtime.max_parallel_tools)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pairs = list(pool.map(one, tool_uses))  # map() keeps result order aligned with the tool_use order
        return [p[0] for p in pairs], [p[1] for p in pairs]
