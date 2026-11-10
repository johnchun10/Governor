"""A feature is a module plus one FEATURES entry: its registrations reach the runtime through build_session."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core import Event, RunView, ToolContext, tool
from agent_core.testing import ScriptedProvider, text, tool_call
from governor.commands import Command
from governor.config import Settings
from governor.features import BUILTIN, FEATURES, Feature
from governor.runtime import build_session
from governor.session import Session

pytestmark = pytest.mark.usefixtures("isolated")


class Says:
    """The one Interaction method a command that only talks needs."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def say(self, text: str, kind: str = "plain") -> None:
        self.lines.append((kind, text))


def test_the_builtin_features_are_the_defaults(isolated: Path):
    Settings(provider="openai").save()
    assert [f.name for f in FEATURES] == ["builtin", "qa", "chats", "map", "budget"]
    session = build_session(isolated, provider_for=lambda info: ScriptedProvider())
    tools = {"read_file", "edit_file", "bash", "task", "map_feature"}
    assert tools <= set(session.runtime.registry.tools)
    providers = [type(p).__name__ for p in session.runtime.context_providers]
    assert providers == ["EnvironmentContext", "AgentsMdContext", "MapContext"]
    assert list(session.commands)[:2] == ["provider", "model"]
    assert list(session.commands)[-4:] == ["qa", "chats", "map", "budget"]
    assert [type(s).__name__ for s in session.sinks.sinks] == ["MapSink", "BudgetSink"]


def test_a_toy_feature_lands_without_touching_anything_else(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    Settings(provider="openai").save()

    @tool(permission="allow")
    def shout(ctx: ToolContext, text: str) -> str:
        """Uppercase text.

        Args:
            text: what to shout.
        """
        return text.upper()

    class Banner:
        def system(self, run: RunView) -> str | None:
            return "<banner>hi</banner>"

        def before_turn(self, run: RunView) -> str | None:
            return None

    class Watch:
        def __init__(self, session: Session) -> None:
            self.session, self.events = session, []

        def handle(self, event: Event) -> None:
            self.events.append(event.type)

    watchers: list[Watch] = []
    closed: list[str] = []

    def sinks(session: Session) -> list[Watch]:
        watchers.append(Watch(session))
        return watchers[-1:]

    toy = Feature(
        "toy",
        tools=lambda project_dir: [shout],
        context_providers=lambda: [Banner()],
        sinks=sinks,
        commands=(Command("toy", "say toy", lambda s, arg: s.say(f"toy {arg}")),),
        status=lambda s: [(f"toy: {len(watchers[0].events)} events", "bold")],
        close=lambda: closed.append("toy"),
    )
    provider = ScriptedProvider([tool_call("shout", {"text": "hi"}), text("done")])
    io = Says()
    session = build_session(isolated, io=io, features=(BUILTIN, toy), provider_for=lambda info: provider)  # type: ignore[arg-type]

    assert "shout" in session.runtime.registry.tools and "toy" in session.commands
    session.submit("/toy now")
    assert io.lines[-1] == ("plain", "toy now")
    assert session.ask("shout hi").ok
    assert "<banner>hi</banner>" in provider.requests[0].system
    assert provider.requests[1].messages[-1].content[0].content == "HI"
    assert watchers[0].session is session and "run_end" in watchers[0].events
    assert watchers[0] in session.sinks.sinks and session.runtime.events.sinks == [session.sinks]
    assert session.status()[-1] == (f"toy: {len(watchers[0].events)} events", "bold")
    session.close()
    assert closed == ["toy"]
