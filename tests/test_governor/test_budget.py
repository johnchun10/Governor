"""/budget: a cap on the session's spend, enforced by a sink that stops the run."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_core import (
    Collector,
    DeltaCallback,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ProviderCancelled,
    StopReason,
    Usage,
)
from agent_core.testing import ScriptedProvider, text, tool_call, tool_calls
from governor.budget import OFF, OTHER, BudgetSink
from governor.config import Settings
from governor.modes.qa import QAMode
from governor.runtime import build_session
from governor.session import Session

from .test_session import ScriptedIO, make_session

pytestmark = pytest.mark.usefixtures("isolated")

# Every scripted call on openai/astra costs 10 in · 5 out tokens: $0.0001 + $0.00025 = $0.00035.
ONE_CALL = 0.00035


def test_a_run_stops_at_the_cap_and_the_turn_says_budget(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([tool_call("list_files", {"path": "."}), text("never sent"), text("after the raise")])
    io = ScriptedIO()
    session = make_session(isolated, provider, io)

    session.submit("/budget 0.0002")
    assert Settings.load().of("budget") == {"cap_usd": 0.0002}
    assert session.status()[-1] == ("budget: $0.0002", "")

    turn = session.ask("look around")
    assert turn.ran and turn.stopped == "budget" and not turn.ok
    assert len(provider.requests) == 1  # the call that crossed the cap was the last; its tool call was refused
    assert session.history is None and session.total.cost_usd == pytest.approx(ONE_CALL)
    sink = session.sinks.find(BudgetSink)
    assert sink is not None and sink.spent == pytest.approx(ONE_CALL) and sink.tripped
    assert "budget: $0.0002 reached" in io.said("dim") and "/budget raises or clears it" in io.said("dim")
    assert session.status()[-1] == ("budget: $0.0002 reached", "warning")

    session.submit("more")  # a run under a spent budget stops before its first model call
    assert len(provider.requests) == 1 and "stopped by budget; the conversation is as it was" in io.said("dim")

    session.submit("/budget 5")
    assert session.status()[-1] == ("budget: $5.00", "")
    assert session.ask("more").ok and len(provider.requests) == 2 and not sink.tripped


def test_a_cap_crossed_by_the_final_reply_keeps_the_finished_turn(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([text("the finished answer"), text("never sent")])
    io = ScriptedIO()
    session = make_session(isolated, provider, io)
    session.submit("/budget 0.0003")

    session.submit("hello")  # one call, $0.00035: the cap is crossed by a reply that ended the run
    assert session.history is not None and [m.text for m in session.history] == ["hello", "the finished answer"]
    assert "budget: $0.0003 reached" in io.said("dim") and "stopped by budget" not in io.said()
    assert "[astra]" in io.said("dim")  # the ordinary status line: the turn was kept and saved
    assert session.chat is not None and session.chat.turns == 1
    assert session.status()[-1] == ("budget: $0.0003 reached", "warning")

    again = session.ask("more")  # the next run stops before its first call
    assert again.stopped == "budget" and len(provider.requests) == 1


def test_the_picker_sets_and_clears_the_cap(isolated: Path):
    io = ScriptedIO(["2", OTHER, "0.75", OFF])
    session = make_session(isolated, ScriptedProvider(), io)

    session.submit("/budget")
    assert Settings.load().of("budget") == {"cap_usd": 2.0} and "budget: $2.00 for this session" in io.said()
    session.submit("/budget")  # "Other amount…", then the typed figure
    assert io.questions[-1] == "Cap in dollars (e.g. 3.50)" and Settings.load().of("budget") == {"cap_usd": 0.75}
    session.submit("/budget")
    assert "budget: off" in io.said() and Settings.load().of("budget") == {} and "budget" not in str(session.status())

    for bad in ("nonsense", "-1", "-0", "0", "nan", "inf", "1e309", "1e308"):
        session.submit(f"/budget {bad}")
    errors = io.said("error")
    assert errors.count("isn't a usable cap; use a positive number of dollars, or off") == 8
    assert Settings.load().features == {}  # an empty namespace isn't saved

    session.submit("/budget 1")
    session.submit("/budget OFF")  # the word is not case-sensitive
    assert Settings.load().features == {} and io.said().endswith("budget: off")


def test_headless_sessions_never_ask(isolated: Path):
    session = make_session(isolated, ScriptedProvider())
    session.submit("/budget")  # no human port: the picker can't open, nothing changes
    assert Settings.load().features == {}
    session.submit("/budget 1")
    assert Settings.load().of("budget") == {"cap_usd": 1.0}


def test_an_unusable_saved_cap_is_ignored_not_fatal(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = make_session(isolated, ScriptedProvider([text("fine")]))
    for value in ("abc", True, float("nan"), -2, [1]):  # a hand-edited settings file
        session.settings.of("budget")["cap_usd"] = value
        assert session.status() and "budget" not in str(session.status())
    assert session.ask("go").ok


@pytest.mark.parametrize("features", [{"budget": 5}, {"budget": [1]}, {"budget": None}, None, [], "x"])
def test_a_misshapen_features_entry_loads_as_empty(isolated: Path, features: object):
    Settings(provider="openai").save()
    settings_file = isolated / "home" / "settings.json"  # GOVERNOR_HOME, set by the isolated fixture
    data = json.loads(settings_file.read_text())
    data["features"] = features
    settings_file.write_text(json.dumps(data))
    loaded = Settings.load()
    assert loaded.features == {} and loaded.of("budget") == {}
    session = build_session(isolated, settings=loaded, provider_for=lambda info: ScriptedProvider())
    assert "budget" not in str(session.status())


def test_esc_after_the_cap_trips_still_counts_as_the_users_stop(isolated: Path):
    session = make_session(isolated, ScriptedProvider(), ScriptedIO())
    session.request_stop("budget")
    session.request_stop()
    assert session._stopped_by == "user"


def test_a_final_reply_cut_off_by_max_tokens_that_crosses_the_cap_is_kept(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cut_off = replace(text("a long answer, cut off"), stop_reason=StopReason.MAX_TOKENS)
    session = make_session(isolated, ScriptedProvider([cut_off]), ScriptedIO())
    session.submit("/budget 0.0003")

    turn = session.ask("explain")
    assert turn.result is not None and turn.result.stop == "max_tokens"
    assert turn.ok and turn.stopped is None and session.history is not None and session.chat is not None


def test_a_cap_that_refuses_the_last_turns_tools_says_budget(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = make_session(isolated, ScriptedProvider([tool_call("list_files", {"path": "."})]), ScriptedIO())
    session.submit("/budget 0.0002")

    # max_turns=1: the one call crosses the cap and asks for a tool, which is refused; the loop then ends on
    # max_turns, and the turn must still say budget rather than look finished.
    turn = session.run(replace(session.spec(), max_turns=1), "look around")
    assert turn.result is not None and turn.result.stop == "max_turns"
    assert turn.stopped == "budget" and not turn.ok and "stopped: budget" in turn.status()


# --- run-tree accounting: Q&A, subagents, a call in flight (from Governor's version in round one) ---------

# One input token costs $0.000001; core computes cost from tokens, not Usage.cost_usd.
MODEL = ModelInfo("openai/budget-test", "openai", "budget-test", input_per_mtok=1.0)


@pytest.fixture
def budget_session(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Session]]:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sessions: list[Session] = []

    def make(provider: ScriptedProvider, io: ScriptedIO | None = None, **kwargs: Any) -> Session:
        session = make_session(isolated, provider, io, resolve_model=lambda name: MODEL, context_providers=[], **kwargs)
        sessions.append(session)
        return session

    yield make
    for session in sessions:
        session.close()


def spent(session: Session) -> float:
    sink = session.sinks.find(BudgetSink)
    assert sink is not None
    return sink.spent


def test_qa_spending_counts_toward_the_same_cap_without_changing_the_conversation(
    budget_session: Callable[..., Session],
):
    provider = ScriptedProvider([text("answer", usage=Usage(100_000)), text("because", usage=Usage(400_000))])
    io = ScriptedIO()
    session = budget_session(provider, io)
    session.submit("/budget 0.5")
    assert session.ask("task").ok
    history = list(session.history or [])
    session.submit("/qa why?")
    assert isinstance(session.mode, QAMode) and session.mode.qa.thread[-1].text == "because"
    assert session.history == history and session.chat is not None and session.chat.turns == 1
    assert spent(session) == pytest.approx(0.5) and session.total.cost_usd == pytest.approx(0.5)
    session.submit("another question")
    assert "stopped by budget" in io.said("dim") and len(provider.requests) == 2
    session.submit("/done")
    assert session.ask("continue task").stopped == "budget" and len(provider.requests) == 2


def test_subagent_calls_count_once_despite_rolled_up_run_end_usage(budget_session: Callable[..., Session]):
    events = Collector()
    provider = ScriptedProvider(
        [
            tool_call("task", {"agent": "explore", "prompt": "child"}, usage=Usage(100_000)),
            text("child report", usage=Usage(300_000)),
            text("combined", usage=Usage(100_000)),
            text("next", usage=Usage(250_000)),
        ]
    )
    session = budget_session(provider, sinks=[events])
    session.submit("/budget 0.75")
    turn = session.ask("delegate")
    assert turn.ok and turn.text == "combined" and len(provider.requests) == 3
    assert turn.usage.cost_usd == pytest.approx(0.5) and spent(session) == pytest.approx(0.5)
    calls = events.of_type("model_call")
    assert [event.depth for event in calls] == [0, 1, 0]
    # The child reports $0.30 and the parent $0.50 on run_end: summing run_end would count the child twice.
    assert sum(event.data["usage"]["cost_usd"] for event in events.of_type("run_end")) == pytest.approx(0.8)
    assert session.ask("use the remaining allowance").ok and spent(session) == pytest.approx(0.75)
    assert session.ask("over the cap").stopped == "budget" and len(provider.requests) == 4


def test_a_subagent_reaching_the_cap_cancels_a_concurrent_call_in_flight(budget_session: Callable[..., Session]):
    waiting = threading.Event()
    cancelled = threading.Event()

    def router(request: ModelRequest) -> ModelResponse:
        prompt = request.messages[0].text
        if prompt == "parent":
            return tool_calls(
                [("task", {"agent": "explore", "prompt": p}) for p in ("spender", "waiting")], usage=Usage(100_000)
            )
        assert prompt == "spender"
        assert waiting.wait(timeout=5), "the other child never entered its model call"
        return text("expensive child", usage=Usage(600_000))

    class CancellableProvider(ScriptedProvider):
        def complete(
            self, request: ModelRequest, *, on_delta: DeltaCallback | None = None, stop: threading.Event | None = None
        ) -> ModelResponse:
            if request.messages[0].text != "waiting":
                return super().complete(request, on_delta=on_delta, stop=stop)
            self.requests.append(request)
            waiting.set()
            assert stop is not None and stop.wait(timeout=5), "the budget never cancelled the call in flight"
            cancelled.set()
            raise ProviderCancelled()

    provider = CancellableProvider(router=router)
    events = Collector()
    session = budget_session(provider, sinks=[events])
    session.submit("/budget 0.5")
    turn = session.ask("parent")
    assert waiting.is_set() and cancelled.is_set()
    assert turn.stopped == "budget" and "stopped: budget" in turn.status()
    assert len(provider.requests) == 3 and len(events.of_type("model_call")) == 2
    assert spent(session) == pytest.approx(0.7)  # the cancelled child reported no usage
    assert turn.usage.cost_usd == pytest.approx(0.7) and session.total.cost_usd == pytest.approx(0.7)
    assert session.history is None and session.chat is None
    assert session.ask("still blocked").stopped == "budget" and len(provider.requests) == 3
