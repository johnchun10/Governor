"""Assemble a core ``Runtime`` and a ``Session`` from Governor's concrete parts.

The two composition roots. ``build_runtime`` is the one place where Governor's
choices meet the core's contracts: which tools, profiles, model registry,
providers, permission policy and context providers a run gets; any of them can
be overridden by keyword. ``build_session`` composes a ``Session`` around it:
settings, the stop event, the permission asker over the human port, the sinks,
the chat store and the command table. Nothing downstream discovers the cwd, a
tty or the environment on its own.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent_core import Event, EventBus, Runtime, Sink, ToolRegistry

from .chats import ChatStore
from .config import Settings
from .context import default_context_providers
from .effort import wrap as with_effort
from .features import FEATURES, Feature
from .models import Selection, resolve
from .permissions import Permissions, asker_from
from .ports import Interaction
from .profiles import load_profiles
from .providers import get_provider
from .session import Session
from .sinks import FanOut
from .tools import default_registry


def build_runtime(
    project_dir: Path,
    selection: Callable[[], Selection] | None = None,
    effort: Callable[[], str | None] | None = None,
    **overrides: Any,
) -> Runtime:
    """``selection`` returns the active provider and models; it is read on every model lookup,
    so a TUI can switch models between turns. ``effort`` returns the effort level to apply
    (``effort.py``), read on every model call. Both default to the saved settings."""
    select = selection or (lambda: Settings.load().selection())
    level = effort or (lambda: Settings.load().effort_level())
    parts: dict[str, Any] = {
        "registry": default_registry(project_dir),
        "profiles": load_profiles(project_dir),
        "resolve_model": lambda name: resolve(name, select(), project_dir),
        "provider_for": lambda info: get_provider(info.provider),
        "permissions": Permissions(),
        "context_providers": default_context_providers(),
    }
    parts.update(overrides)
    parts["provider_for"] = with_effort(parts["provider_for"], level)
    return Runtime(**parts)


def build_session(
    project_dir: Path,
    *,
    io: Interaction | None = None,
    sinks: Sequence[Sink] = (),
    settings: Settings | None = None,
    features: Sequence[Feature] = FEATURES,
    save_chats: bool = True,
    trace_label: str = "session",
    **overrides: Any,
) -> Session:
    """A ``Session`` over a runtime built here, with every feature's registrations installed.

    ``io`` is the surface's human port (None: headless, never asks); ``sinks`` receive every run
    event; ``overrides`` go to ``build_runtime`` and win over the features' tools and context
    providers. The trace file is ``.governor/runs/<time>-<trace_label>.jsonl``, opened at the first run.

    The event bus gets one ``FanOut`` (``session.sinks``): the surface's ``sinks``, every
    feature's and the trace subscribe through it, so a sink that raises is reported once
    (through ``io``, or as a logging warning when headless) and never ends a run.
    """
    settings = settings or Settings.load()
    stop = threading.Event()
    report = (lambda sink, event, exc: session.say(_sink_failed(sink, event, exc), "error")) if io is not None else None
    fanout = FanOut(sinks, on_error=report)
    parts: dict[str, Any] = {
        "registry": ToolRegistry().register(*(t for f in features for t in f.tools(project_dir))),
        "context_providers": [p for f in features for p in f.context_providers()],
        "events": EventBus([fanout]),
        "permissions": Permissions(asker=asker_from(io, stop)),
        "stop": stop,
        **overrides,
    }
    runtime = build_runtime(project_dir, selection=settings.selection, effort=settings.effort_level, **parts)
    if runtime.events.sinks != [
        fanout
    ]:  # an ``events`` override brought its own bus: its sinks go through the fan-out too
        for sink in runtime.events.sinks:
            if sink is not fanout:
                fanout.add(sink)
        runtime.events.sinks[:] = [fanout]
    session = Session(
        project_dir,
        settings,
        runtime,
        io,
        [c for f in features for c in f.commands],
        ChatStore(project_dir) if save_chats else None,
        sinks=fanout,
        status=[f.status for f in features],
        trace_label=trace_label,
        closers=[f.close for f in features if f.close is not None],
    )
    for feature in features:
        for sink in feature.sinks(session):
            fanout.add(sink)
    return session


def _sink_failed(sink: Sink, event: Event, exc: Exception) -> str:
    return (
        f"{type(sink).__name__} failed on a {event.type} event: {exc}. The run continues; "
        "this sink still gets later events, and further failures of it are not reported"
    )
