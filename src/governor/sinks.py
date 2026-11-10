"""Event sinks: where Governor sends run events (the trace file, a console), and the
``FanOut`` they all hang off."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO, TypeVar

from agent_core import Event, Sink

log = logging.getLogger(__name__)

S = TypeVar("S")

OnError = Callable[[Sink, Event, Exception], None]
"""Told once per subscriber about its first failure: the sink, the event it failed on, and the error."""


class FanOut:
    """One sink over many: every subscriber gets every event, and one that raises is skipped
    for that event, never ending the run.

    Core's ``EventBus`` calls its sinks with no isolation, so a bug in a sink that does
    work per event (a learning sink, a UI mid-shutdown, a full disk under the trace) would
    end ``Agent.run`` with a traceback. ``build_session`` hands the bus this one sink and
    registers everything else here: the surface's sinks, every ``Feature.sinks``, the
    session's trace. A subscriber's first failure goes to ``on_error`` (default: a logging
    warning); later ones are silent, so a sink raising on every ``delta`` reports once.
    """

    def __init__(self, sinks: Iterable[Sink] = (), on_error: OnError | None = None) -> None:
        self.sinks: list[Sink] = list(sinks)
        self.on_error: OnError = on_error or _warn
        self._failed: set[int] = set()

    def add(self, sink: Sink) -> None:
        self.sinks.append(sink)

    def remove(self, sink: Sink) -> None:
        self.sinks.remove(sink)
        self._failed.discard(id(sink))

    def find(self, kind: type[S]) -> S | None:
        """The first subscriber of type ``kind``: how a feature's command reaches the sink it registered."""
        return next((s for s in self.sinks if isinstance(s, kind)), None)

    def handle(self, event: Event) -> None:
        for sink in list(self.sinks):
            try:
                sink.handle(event)
            except Exception as exc:  # isolating subscribers is the point
                if id(sink) not in self._failed:
                    self._failed.add(id(sink))
                    self._report(sink, event, exc)

    def _report(self, sink: Sink, event: Event, exc: Exception) -> None:
        """Tell ``on_error``; if telling fails too (a UI shutting down), log it, and never let it end the run."""
        try:
            self.on_error(sink, event, exc)
        except Exception:
            with contextlib.suppress(Exception):
                _warn(sink, event, exc)


def _warn(sink: Sink, event: Event, exc: Exception) -> None:
    log.warning("%s failed on a %s event: %s", type(sink).__name__, event.type, exc, exc_info=exc)


class JsonlTrace:
    """Append every event to a JSONL file (one file per session).

    Streamed ``delta`` events are left out: the complete message follows in
    ``model_call``, and a token per line would swamp the trace.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def handle(self, event: Event) -> None:
        if event.type == "delta":
            return
        self._fh.write(event.to_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _short(value: Any, limit: int = 80) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[: limit - 1] + "…"


FILE_TOOLS = ("read_file", "write_file", "edit_file")
"""Tools whose ``path`` argument the run summary lists, so it says what a run looked at and changed."""


def summarize_calls(calls: list[tuple[str, dict[str, Any]]], limit: int = 6) -> str:
    """One line for a run's tool calls: counts per tool, then the files read and written.

    ``17 tool calls: 13 read_file, 4 list_files · read AGENTS.md, README.md, +7 more``
    """
    if not calls:
        return "no tool calls"
    counts: dict[str, int] = {}
    for name, _ in calls:
        counts[name] = counts.get(name, 0) + 1
    parts = [
        f"{len(calls)} tool call{'s' if len(calls) != 1 else ''}: " + ", ".join(f"{n} {t}" for t, n in counts.items())
    ]
    for verb, tools in (("read", ("read_file",)), ("changed", ("write_file", "edit_file"))):
        paths = list(dict.fromkeys(str(a.get("path")) for t, a in calls if t in tools and a.get("path")))
        if paths:
            more = f", +{len(paths) - limit} more" if len(paths) > limit else ""
            parts.append(f"{verb} " + ", ".join(paths[:limit]) + more)
    return " · ".join(parts)


class ConsolePrinter:
    """Human-readable progress lines, indented by subagent depth: subagent starts and ends, one
    summary of each run's tool calls when it ends, errors, and (``verbose``) every tool call as it
    happens, tool output and intermediate model text. The top-level reply itself is not printed;
    the TUI streams it from ``delta`` events."""

    DIM, BOLD, RED, CYAN, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[36m", "\033[0m"

    def __init__(
        self, stream: TextIO = sys.stderr, verbose: bool = False, write: Callable[[str], None] | None = None
    ) -> None:
        """``write`` replaces the stream (the TUI routes lines through its log)."""
        self.stream = stream
        self.verbose = verbose
        self.write = write
        self.color = write is not None or stream.isatty()
        self._calls: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        """run_id -> its tool calls so far, summarized when the run ends."""

    def _c(self, code: str, s: str) -> str:
        return f"{code}{s}{self.RESET}" if self.color else s

    def _w(self, depth: int, s: str) -> None:
        line = "  " * depth + s
        if self.write is not None:
            self.write(line)
            return
        self.stream.write(line + "\n")
        self.stream.flush()

    def handle(self, e: Event) -> None:
        d = e.depth
        if e.type == "run_start" and e.depth > 0:
            self._w(d, self._c(self.CYAN, f"↳ subagent [{e.agent}] ({e.data['model']}): {_short(e.data['task'], 100)}"))
        elif e.type == "model_call":
            text = "".join(b.get("text", "") for b in e.data["message"]["content"] if b["type"] == "text").strip()
            if text and (e.depth > 0 or e.data["stop_reason"] == "tool_use" or self.verbose):
                self._w(d, self._c(self.DIM, _short(text, 300)) if e.depth > 0 else text)
        elif e.type == "tool_call":
            self._calls.setdefault(e.run_id, []).append((e.data["name"], e.data["input"]))
            if self.verbose:
                args = ", ".join(f"{k}={_short(v, 60)}" for k, v in e.data["input"].items())
                self._w(d, self._c(self.BOLD, f"⏺ {e.data['name']}") + f"({args})")
        elif e.type == "tool_result" and (e.data["is_error"] or self.verbose):
            color = self.RED if e.data["is_error"] else self.DIM
            prefix = "⎿ " if self.verbose else f"⎿ {e.data['name']}: "
            self._w(d + 1, self._c(color, prefix + _short(e.data["content"], 200)))
        elif e.type == "run_end":
            if calls := self._calls.pop(e.run_id, None):
                self._w(d, self._c(self.BOLD, "⏺ ") + self._c(self.DIM, summarize_calls(calls)))
            if e.depth > 0:
                u = e.data["usage"]
                self._w(d, self._c(self.CYAN, f"✓ [{e.agent}] done ({e.data['turns']} turns, ${u['cost_usd']:.4f})"))
        elif e.type == "error":
            self._w(d, self._c(self.RED, f"✗ {e.data['message']}"))
