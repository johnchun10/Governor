"""Built-in tools and the default registry."""

from __future__ import annotations

from pathlib import Path

from agent_core import Tool, ToolRegistry, task_tool

from .external import load_external_tools
from .files import FILE_TOOLS
from .shell import bash

__all__ = ["builtin_tools", "default_registry"]


def builtin_tools(project_dir: Path | None = None) -> list[Tool]:
    """Built-in tools plus any external tools under ``<project>/.governor/tools`` (the builtin feature's)."""
    tools: list[Tool] = [*FILE_TOOLS, bash, task_tool]
    if project_dir is not None:
        tools += load_external_tools(project_dir / ".governor" / "tools")
    return tools


def default_registry(project_dir: Path | None = None) -> ToolRegistry:
    return ToolRegistry().register(*builtin_tools(project_dir))
