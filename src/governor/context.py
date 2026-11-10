"""Built-in context providers (environment info, AGENTS.md)."""

from __future__ import annotations

import platform
from datetime import date

from agent_core import ContextProvider, RunView


class EnvironmentContext:
    """Working directory, platform and date."""

    def system(self, run: RunView) -> str | None:
        return (
            "<environment>\n"
            f"working directory: {run.cwd}\n"
            f"platform: {platform.system()} {platform.release()}\n"
            f"date: {date.today().isoformat()}\n"
            "</environment>"
        )

    def before_turn(self, run: RunView) -> str | None:
        return None


class AgentsMdContext:
    """Project instructions from AGENTS.md (searched from cwd upward)."""

    def __init__(self, filename: str = "AGENTS.md") -> None:
        self.filename = filename

    def system(self, run: RunView) -> str | None:
        for directory in [run.cwd, *run.cwd.parents]:
            path = directory / self.filename
            if path.is_file():
                return f'<project_instructions source="{path}">\n{path.read_text().strip()}\n</project_instructions>'
        return None

    def before_turn(self, run: RunView) -> str | None:
        return None


def default_context_providers() -> list[ContextProvider]:
    return [EnvironmentContext(), AgentsMdContext()]
