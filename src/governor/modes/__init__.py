"""Input modes: what a typed line means while one is on.

``/qa`` turns the input box into questions about the conversation; another
feature with its own input would do the same. A mode
owns all of that: its sub-commands (``model``, ``done``), the line handler, the
placeholder, the status badge, the hint explained once at entry, and the input
box's title. ``Session.mode`` holds the active one and ``Session.submit`` routes
through it; a surface reads ``session.mode`` and never tests a feature flag.
Views stay the surface's job: ``Interaction.show_mode`` shows or removes a
mode's panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..commands import Command
    from ..session import Session


class Mode(Protocol):
    name: str
    """Short label for the status bar while the mode is on (``Q&A``)."""
    placeholder: str
    """What the empty input box says."""
    hint: str
    """Explained once, at entry: the first line is the panel's heading, the rest are notes."""
    commands: dict[str, Command]
    """Sub-commands that exist only while the mode is on; they shadow the session's."""

    def submit(self, session: Session, line: str) -> None:
        """Handle a typed line that is not a slash command."""
        ...

    def leave(self, session: Session) -> None:
        """The mode is being left (Esc, ``/done``, ``/clear``, a resumed chat)."""
        ...

    def title(self, draft: str) -> str | None:
        """What the input box's title says while ``draft`` is being typed, or None for no title."""
        ...
