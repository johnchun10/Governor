"""Core tests use only agent_core: tiny in-memory tools and a scripted provider."""

from __future__ import annotations

import pytest

from agent_core import (
    AgentSpec,
    Collector,
    EventBus,
    ModelInfo,
    StaticPolicy,
    ToolContext,
    ToolError,
    ToolRegistry,
    task_tool,
    tool,
)
from agent_core.testing import ScriptedProvider

NOTES: dict[str, str] = {}


@tool(permission="allow")
def read_note(ctx: ToolContext, name: str) -> str:
    """Read a note.

    Args:
        name: note name.
    """
    if name not in NOTES:
        raise ToolError(f"no note named {name}")
    return NOTES[name]


@tool(permission="ask")
def write_note(ctx: ToolContext, name: str, text: str) -> str:
    """Write a note.

    Args:
        name: note name.
        text: note text.
    """
    NOTES[name] = text
    return "saved"


@tool(permission="allow")
def noop(ctx: ToolContext) -> str:
    """Do nothing."""
    return "ok"


def fake_model(name: str | None) -> ModelInfo:
    return ModelInfo(name or "fake", "fake", f"fake-{name or 'default'}", input_per_mtok=1.0, output_per_mtok=2.0)


@pytest.fixture(autouse=True)
def _clear_notes():
    NOTES.clear()
    yield


@pytest.fixture
def make_runtime():
    from agent_core import Runtime

    def _make(provider: ScriptedProvider, *, profiles: dict[str, AgentSpec] | None = None, policy=None, **kw):
        collector = Collector()
        runtime = Runtime(
            registry=ToolRegistry().register(read_note, write_note, noop, task_tool),
            profiles=profiles or {"general": AgentSpec("general", "full tools")},
            resolve_model=fake_model,
            provider_for=lambda info: provider,
            events=EventBus([collector]),
            permissions=policy or StaticPolicy(yolo=True),
            **kw,
        )
        return runtime, collector

    return _make
