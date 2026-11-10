"""Permission policy interface.

The loop asks the runtime's policy before running every tool call. Interactive
prompting, env-var overrides and so on are governor concerns (see
``governor.permissions``); core only defines the contract and a safe default.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

Decision = Literal["allow", "ask", "deny"]


class PermissionPolicy(Protocol):
    def check(
        self, tool: str, args: dict[str, Any], default: Decision, profile: dict[str, Decision] | None = None
    ) -> str | None:
        """Return None if the call may proceed, else the refusal message for the model.

        ``default`` is the tool's own default; ``profile`` holds the agent
        profile's per-tool overrides.
        """
        ...


class StaticPolicy:
    """Non-interactive policy: ``ask`` is treated as ``deny`` unless ``yolo`` is set."""

    def __init__(self, overrides: dict[str, Decision] | None = None, *, yolo: bool = False) -> None:
        self.overrides = dict(overrides or {})
        self.yolo = yolo

    def check(
        self, tool: str, args: dict[str, Any], default: Decision, profile: dict[str, Decision] | None = None
    ) -> str | None:
        decision = self.overrides.get(tool) or (profile or {}).get(tool) or ("allow" if self.yolo else default)
        if decision == "allow":
            return None
        return f"permission denied: {tool} is not allowed by policy"
