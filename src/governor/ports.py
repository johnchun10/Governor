"""The port that faces the human.

Run events leave through core's ``Sink``. Everything that comes *from* the human
while the application layer works (a picker, a permission prompt, a sign-in
question, a line to show) goes through ``Interaction``, implemented once per
surface: the TUI's modal dialogs, a test's scripted answers, later a web
client's question queue. Application code (``session.py``, ``commands.py``,
``modes/``) talks to the human only through it, never through a widget,
``/dev/tty`` or ``isatty``; a headless caller passes ``io=None`` and the session
runs without asking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from .modes import Mode

Kind = Literal["plain", "dim", "error", "user", "answer"]
"""How a line reads: ordinary, de-emphasised, an error, the human's own input, or a model's reply."""


@dataclass(frozen=True)
class Tab:
    """One page of a tabbed picker (``choose_tabbed``): its options and the current choice."""

    id: str
    label: str
    options: list[tuple[str, str]]
    default: str | None = None


class Interaction(Protocol):
    """One surface's way of showing lines and asking questions. The blocking calls run on a worker thread."""

    def say(self, text: str, kind: Kind = "plain") -> None: ...

    def text(self, message: str, default: str = "", *, suggestions: list[str] | None = None) -> str: ...

    def secret(self, message: str) -> str: ...

    def choose(self, message: str, options: list[tuple[str, str]], default: str | None = None) -> str | None: ...

    def choose_tabbed(self, message: str, tabs: Sequence[Tab], start: str) -> tuple[str, str] | None:
        """Pick from one of several pages (←/→ switch). Returns ``(tab id, value)``, or None when cancelled."""
        ...

    def confirm(self, message: str, *, yes: str = "Yes", no: str = "No") -> bool: ...

    def run_interactive(self, command: list[str]) -> int:
        """Hand the real terminal to an interactive command (``aws login``). Returns its exit code."""
        ...

    def show_mode(self, mode: Mode | None) -> None:
        """A mode was entered (``mode``) or left (``None``): show or remove its panel, and return once that is done."""
        ...

    def clear(self) -> None:
        """Forget what was shown; the next line starts a fresh log."""
        ...
