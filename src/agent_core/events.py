"""Run events: the trace schema every component reads and writes.

Every run emits a flat stream of events tagged with ``run_id`` and
``parent_run_id``, so a whole tree of agents and subagents lands in one
JSONL trace. The trace is the record used for replay, evaluation and cost
accounting. It also serves as the language-neutral interface: a
TypeScript UI can render the event stream without importing Python.

Changing an event's fields is a contract change: bump ``SCHEMA_VERSION``.

Event types and their ``data`` payloads:

* ``run_start``: agent, model, task, system, tools, depth
* ``delta``: turn, kind (text / reasoning / tool_input), text, index. One per
  streamed fragment while a model call runs; the complete message follows in
  ``model_call``, so traces may drop deltas (Governor's JSONL trace does)
* ``model_call``: turn, message (assistant), stop_reason, usage, latency_s
* ``context``: turn, text (dynamic context injected before a turn)
* ``tool_call``: id, name, input
* ``tool_result``: id, name, content, is_error, outcome (ok / invalid / denied / error), duration_s
* ``run_end``: stop, text, usage (including subagents), turns, duration_s
* ``error``: message

Schema history: v2 added ``delta``, ``tool_result.outcome`` and the ``run_end.stop``
values ``cancelled``, ``invalid_output`` and ``invalid_tool_calls``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Event:
    type: str
    run_id: str
    agent: str
    depth: int = 0
    parent_run_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


class Sink(Protocol):
    def handle(self, event: Event) -> None: ...


class EventBus:
    """Fan events out to sinks. Thread-safe: parallel subagents share one bus."""

    def __init__(self, sinks: list[Sink] | None = None) -> None:
        self.sinks: list[Sink] = list(sinks or [])
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        with self._lock:
            for sink in self.sinks:
                sink.handle(event)


class Collector:
    """Keep events in memory (tests, programmatic use)."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def handle(self, event: Event) -> None:
        self.events.append(event)

    def of_type(self, type_: str) -> list[Event]:
        return [e for e in self.events if e.type == type_]
