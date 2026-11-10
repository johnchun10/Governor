"""Tool permissions: allow / ask / deny, decided per tool name.

Precedence, highest first: an explicit override (``GOVERNOR_ALLOW`` /
``GOVERNOR_DENY``, or "always" answered at a prompt), then the agent profile,
then yolo mode (``/yolo`` in the TUI), then the tool's own default. Read-only tools default to ``allow``; anything that
writes or executes defaults to ``ask``.

Asking goes through the session's human port (``asker_from``). Without one (a
headless session) an ``ask`` is refused with a hint to use
``GOVERNOR_ALLOW``; nothing here reads the terminal.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent_core import Decision

if TYPE_CHECKING:
    from .ports import Interaction

Asker = Callable[[str, dict[str, Any]], str]
"""Answers 'y', 'n' or 'a' (yes, and don't ask again for this tool)."""


def asker_from(io: Interaction | None, stop: threading.Event | None = None) -> Asker | None:
    """An asker that puts each question to ``io``, or None (never ask) without one. Declines once ``stop`` is set."""
    if io is None:
        return None

    def ask(tool: str, args: dict[str, Any]) -> str:
        if stop is not None and stop.is_set():
            return "n"
        preview = ", ".join(f"{k}={str(v)[:200]!r}" for k, v in args.items())
        answer = io.choose(
            f"Allow {tool}({preview})?",
            [("y", "Yes"), ("a", f"Yes, and don't ask again for {tool} this session"), ("n", "No")],
        )
        return answer or "n"

    return ask


class Permissions:
    def __init__(
        self,
        overrides: dict[str, Decision] | None = None,
        *,
        yolo: bool = False,
        asker: Asker | None = None,
    ) -> None:
        self.overrides: dict[str, Decision] = {}
        allow = [n.strip() for n in os.environ.get("GOVERNOR_ALLOW", "").split(",") if n.strip()]
        for name in allow:
            self.overrides[name] = "allow"
        for name in filter(None, os.environ.get("GOVERNOR_DENY", "").split(",")):
            self.overrides[name.strip()] = "deny"
        self.overrides.update(overrides or {})
        self.yolo = yolo or "*" in allow  # GOVERNOR_ALLOW=* allows everything
        self.asker = asker
        self._lock = threading.Lock()  # parallel tool calls must not prompt at once

    def decide(self, tool: str, default: Decision, profile: dict[str, Decision] | None = None) -> Decision:
        if tool in self.overrides:
            return self.overrides[tool]
        if profile and tool in profile:
            return profile[tool]
        if self.yolo:
            return "allow"
        return default

    def check(
        self, tool: str, args: dict[str, Any], default: Decision, profile: dict[str, Decision] | None = None
    ) -> str | None:
        """Return None if the call may proceed, else a reason it was refused."""
        decision = self.decide(tool, default, profile)
        if decision == "allow":
            return None
        if decision == "deny":
            return f"permission denied: {tool} is disabled by policy"
        if self.asker is None:
            return f"permission denied: {tool} needs approval but no one can be asked (set GOVERNOR_ALLOW={tool}, or GOVERNOR_ALLOW=* for all)"
        with self._lock:
            if self.overrides.get(tool) == "allow":  # approved with 'a' while we waited
                return None
            answer = self.asker(tool, args)
            if answer == "a":
                self.overrides[tool] = "allow"
                return None
            if answer == "y":
                return None
        return f"permission denied: the user declined this {tool} call"
