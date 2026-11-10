"""Context provider interface: the seam where knowledge enters a run.

A provider has two hooks:

* ``system(run)`` is called once when a run starts, and its text is appended to
  the system prompt. Put stable content here (environment, AGENTS.md), since
  it is part of the cached prompt prefix.
* ``before_turn(run)`` is called before every model call. Its text is added to
  the newest user message as a ``<context>`` block. Use this for dynamic
  context such as retrieved memories, filtered previews or recall hints.
  Earlier messages are never edited (history stays append-only), so the
  cached prefix survives and the trace shows exactly what was injected and
  when.

Return ``None`` from either hook to add nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .types import Message

if TYPE_CHECKING:
    from .agent import AgentSpec


@dataclass(frozen=True)
class RunView:
    """Read-only view of a run, as seen by context providers."""

    spec: AgentSpec
    task: str
    cwd: Path
    depth: int
    turn: int
    messages: tuple[Message, ...]


class ContextProvider(Protocol):
    def system(self, run: RunView) -> str | None: ...

    def before_turn(self, run: RunView) -> str | None: ...
