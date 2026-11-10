"""Features: one registration point each, installed by ``build_session``.

A feature is a module plus registrations: the tools it adds, the context
providers it injects, the sinks that watch its runs, its global command (one
per feature; sub-commands belong in its ``Mode``), its status-bar fragments
(shown only while it is on), and what to release when the session closes.
``FEATURES`` lists them in the order their commands appear. Adding a feature is
one module and one entry here, never an edit to the TUI's dispatch, status or
escape code. Nothing is discovered: no entry points, no plugin directory.

A feature's settings live under its name in the saved settings
(``session.settings.of("budget")``), never as a field on ``Settings``. A mode is
entered by the feature's command rather than installed here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agent_core import ContextProvider, Sink, Tool

from . import budget, chats
from . import map as project_map
from .commands import BUILTIN as BUILTIN_COMMANDS
from .commands import Command
from .context import default_context_providers
from .modes import qa
from .tools import builtin_tools

if TYPE_CHECKING:
    from .session import Session


def _no_tools(project_dir: Path) -> Sequence[Tool]:
    return ()


def _no_context_providers() -> Sequence[ContextProvider]:
    return ()


def _no_sinks(session: Session) -> Sequence[Sink]:
    return ()


def _no_status(session: Session) -> Sequence[tuple[str, str]]:
    return ()


@dataclass(frozen=True)
class Feature:
    name: str
    tools: Callable[[Path], Sequence[Tool]] = _no_tools
    """Tools to register, given the project directory (external tools live under it)."""
    context_providers: Callable[[], Sequence[ContextProvider]] = _no_context_providers
    sinks: Callable[[Session], Sequence[Sink]] = _no_sinks
    """Sinks that watch every run, built once the session exists (a learning sink needs its store)."""
    commands: tuple[Command, ...] = ()
    status: Callable[[Session], Sequence[tuple[str, str]]] = _no_status
    """Status-bar fragments as (text, style) like ``Session.status()``; empty unless the feature is on."""
    close: Callable[[], None] | None = None
    """Release what the feature holds open (server processes, files) when the session closes."""


BUILTIN = Feature(
    "builtin", tools=builtin_tools, context_providers=default_context_providers, commands=BUILTIN_COMMANDS
)
"""The harness itself: file and shell tools, subagents, external tools, the environment and AGENTS.md
context, and the model/provider/sign-in/agent/effort/conversation commands."""

QA = Feature("qa", commands=(Command("qa", "ask about this conversation without adding to it", qa.enter, echo=False),))

CHATS = Feature("chats", commands=(Command("chats", "list saved chats and resume one", chats.resume_command),))

MAP = Feature(
    "map",
    tools=project_map.tools,
    context_providers=project_map.context_providers,
    sinks=project_map.sinks,
    commands=(project_map.COMMAND,),
)
"""The project map (``.governor/map/``): area and feature cards in the system prompt, the ``map_feature``
tool, the sink that confirms the paths a run touched, and ``/map``."""

BUDGET = Feature(
    "budget",
    sinks=budget.sinks,
    commands=(Command("budget", "cap what this session spends (a run stops at the cap)", budget.command, budget.args),),
    status=budget.status,
)

FEATURES: tuple[Feature, ...] = (BUILTIN, QA, CHATS, MAP, BUDGET)
