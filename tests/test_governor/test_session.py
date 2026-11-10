"""The Session: the conversation and the one run sequence, driven headlessly (scripted model, scripted human)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from agent_core import Event, EventBus, Message, ModelRequest, ProviderError, ReasoningBlock, TextBlock
from agent_core.testing import ScriptedProvider, text, tool_call
from governor.chats import Chat, ChatStore
from governor.config import Settings
from governor.features import FEATURES, Feature
from governor.modes.qa import QAMode
from governor.ports import Kind
from governor.runtime import build_session
from governor.session import Session
from governor.sinks import JsonlTrace

pytestmark = pytest.mark.usefixtures("isolated")


class ScriptedIO:
    """Answers prompts from a list and records what was said and asked."""

    def __init__(self, answers: list[str] = ()) -> None:
        self.answers = list(answers)
        self.lines: list[tuple[str, str]] = []
        self.questions: list[str] = []
        self.modes: list[object] = []
        self.cleared = 0

    def say(self, text: str, kind: Kind = "plain") -> None:
        self.lines.append((kind, text))

    def text(self, message: str, default: str = "", *, suggestions: list[str] | None = None) -> str:
        self.questions.append(message)
        return self.answers.pop(0) or default

    def secret(self, message: str) -> str:
        self.questions.append(message)
        return self.answers.pop(0)

    def choose(self, message: str, options: list[tuple[str, str]], default: str | None = None) -> str | None:
        self.questions.append(message)
        answer = self.answers.pop(0)
        assert answer in [value for value, _ in options], (answer, options)
        return answer

    def choose_tabbed(self, message: str, tabs, start: str) -> tuple[str, str] | None:
        self.questions.append(message)
        tab, _, value = self.answers.pop(0).partition(":")
        assert value in [v for t in tabs if t.id == tab for v, _ in t.options], (tab, value)
        return tab, value

    def confirm(self, message: str, *, yes: str = "Yes", no: str = "No") -> bool:
        self.questions.append(message)
        return self.answers.pop(0) == "yes"

    def run_interactive(self, command: list[str]) -> int:
        return 0

    def show_mode(self, mode: object) -> None:
        self.modes.append(mode)

    def clear(self) -> None:
        self.cleared += 1

    def said(self, kind: str | None = None) -> str:
        return "\n".join(t for k, t in self.lines if kind is None or k == kind)


def make_session(project: Path, provider: ScriptedProvider, io: ScriptedIO | None = None, **kwargs) -> Session:
    Settings(provider="openai").save()
    return build_session(project, io=io, provider_for=lambda info: provider, **kwargs)


def test_ask_continues_the_conversation_and_saves_a_chat(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([text("one"), text("two")])
    session = make_session(isolated, provider, trace_label="t")

    first = session.ask("first")
    assert first.ok and first.text == "one" and first.model is not None and first.model.alias == "openai/astra"
    assert first.status().startswith("[astra] 10 in / 5 out tokens · $")
    session.ask("second")
    assert session.history is not None and [m.text for m in session.history] == ["first", "one", "second", "two"]
    assert provider.requests[1].messages[:2] == (Message.user("first"), text("one").message)
    assert session.total.input_tokens == 20
    assert session.conversation is not None and session.conversation[1].alias == "openai/astra"
    [chat] = ChatStore(isolated).list()
    assert chat.title == "first" and chat.turns == 2 and chat.usage.input_tokens == 20
    assert session.trace is not None and session.trace.name.endswith("-t.jsonl") and session.trace.is_file()


def test_stopped_and_failed_runs_leave_the_conversation(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sessions: list[Session] = []

    def stopping(request: ModelRequest):  # the user presses Esc while the model is answering
        sessions[0].request_stop()
        return text("never shown")

    def failing(request: ModelRequest):
        raise ProviderError("boom", hint="try again")

    provider = ScriptedProvider([text("kept"), stopping, failing])
    sessions.append(make_session(isolated, provider))
    session = sessions[0]
    assert session.ask("keep").ok
    stopped = session.ask("stop me")
    assert stopped.ran and stopped.stopped and not stopped.ok
    failed = session.ask("fail")
    assert failed.ran and failed.error is not None and "boom" in failed.error and not failed.ok
    assert session.history is not None and [m.text for m in session.history] == ["keep", "kept"]
    assert ChatStore(isolated).list()[0].turns == 1
    assert session.total.input_tokens == 10  # a cancelled call reports no usage; the failed one none either


def test_headless_session_never_asks(isolated: Path):
    # No key in the environment and no human port: the run goes ahead (the provider would report),
    # and a tool that needs approval is refused instead of prompting.
    provider = ScriptedProvider([text("hello there"), tool_call("bash", {"command": "ls"}), text("done")])
    session = make_session(isolated, provider)
    assert session.ask("hi").ok
    assert session.ask("run ls").ok
    refused = provider.requests[2].messages[-1].content[0]
    assert refused.type == "tool_result" and refused.is_error and "no one can be asked" in refused.content


def test_sign_in_is_offered_through_the_interaction(isolated: Path):
    provider = ScriptedProvider([text("hi")])
    io = ScriptedIO(["no"])
    session = make_session(isolated, provider, io)
    turn = session.ask("hello")
    assert not turn.ran and turn.error is None and not provider.requests
    assert io.questions == ["You're not signed in to openai. Sign in now?"]


def test_permission_prompt_goes_through_the_interaction(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([tool_call("bash", {"command": "echo hi"}), text("ran")])
    io = ScriptedIO(["y"])
    session = make_session(isolated, provider, io)
    assert session.ask("echo").ok
    assert io.questions == ["Allow bash(command='echo hi')?"]
    result = provider.requests[1].messages[-1].content[0]
    assert result.type == "tool_result" and not result.is_error


def test_resume_strips_reasoning_when_the_provider_changed(isolated: Path):
    thinking = ReasoningBlock(raw={"type": "thinking", "signature": "x"}, provider="anthropic")
    chat = Chat.new("old task", "anthropic", "anthropic/opus", "general")
    chat.messages = [Message.user("old task"), Message("assistant", (thinking, TextBlock("old answer")))]
    ChatStore(isolated).save(chat)
    session = make_session(isolated, ScriptedProvider())

    resumed = session.resume(chat)
    assert resumed.switched and "now on openai" in resumed.note
    assert [b.type for m in resumed.messages for b in m.content] == ["text", "text"]
    assert session.history == resumed.messages and session.chat is chat
    assert session.conversation is not None and session.conversation[1].provider == "openai"


def test_submit_routes_commands_modes_and_tasks(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([text(r) for r in ("built it", "**because**", "cheap answer", "back to work")])
    io = ScriptedIO()
    session = make_session(isolated, provider, io)

    session.submit("/nonsense")
    assert "unknown command /nonsense" in io.said("error")
    session.submit("build it")
    assert session.history is not None and [m.text for m in session.history] == ["build it", "built it"]

    session.submit("/qa why?")
    mode = session.mode
    assert isinstance(mode, QAMode) and io.modes == [mode]
    question = provider.requests[1]
    assert question.messages[:2] == (Message.user("build it"), text("built it").message)
    assert question.messages[2].text.startswith("<qa>") and question.messages[2].text.endswith("why?")
    assert mode.qa.thread[-1].text == "**because**"
    assert [m.text for m in session.history] == ["build it", "built it"]  # nothing added

    session.submit("/model luna")  # the mode's /model, not the session's
    assert Settings.load().selection().roles == {"main": "astra", "subagent": "luna", "qa": "luna"}
    session.submit("what else?")
    assert provider.requests[2].model == "gpt-6-luna"
    session.submit("/done")
    assert session.mode is None and io.modes == [mode, None]

    session.submit("continue")
    assert [m.text for m in session.history] == ["build it", "built it", "continue", "back to work"]
    users = io.said("user")
    assert "? why?" in users and "? what else?" in users and "› continue" in users
    assert "/qa why?" not in io.said() and "› /model luna" in users
    assert "[astra]" in io.said("dim") and "[luna]" in io.said("dim")


def test_clear_forgets_the_conversation_and_leaves_the_mode(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    io = ScriptedIO()
    session = make_session(isolated, ScriptedProvider([text("a")]), io)
    session.submit("task")
    session.submit("/qa")
    assert session.mode is not None
    session.submit("/clear")
    assert session.history is None and session.chat is None and session.mode is None
    assert io.cleared == 1 and io.modes[-1] is None and "new conversation" in io.said("dim")


def test_close_releases_the_trace(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = make_session(isolated, ScriptedProvider([text("one")]))
    session.ask("first")
    trace = session.trace
    session.close()
    assert trace is not None and session.trace == trace
    types = [json.loads(line)["type"] for line in trace.read_text().splitlines()]
    assert types == ["run_start", "model_call", "run_end"]
    assert session.sinks.find(JsonlTrace) is None


def test_a_raising_sink_never_ends_the_run(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class Broken:
        def handle(self, event: Event) -> None:
            raise RuntimeError("disk full")

    broken = Feature("broken", sinks=lambda session: [Broken()])
    provider = ScriptedProvider([tool_call("list_files", {"path": "."}), text("fine")])
    io = ScriptedIO()
    session = make_session(isolated, provider, io, features=(*FEATURES, broken))

    turn = session.ask("look around")
    assert turn.ok and turn.text == "fine"
    # The bus holds one FanOut and everything subscribes through it; the trace, registered after the
    # broken sink, still saw every event, and the failure was reported once rather than per event.
    assert session.runtime.events.sinks == [session.sinks]
    assert session.sinks.find(Broken) is not None and session.sinks.find(JsonlTrace) is not None
    assert session.trace is not None
    types = [json.loads(line)["type"] for line in session.trace.read_text().splitlines()]
    assert types == ["run_start", "model_call", "tool_call", "tool_result", "model_call", "run_end"]
    errors = [line for kind, line in io.lines if kind == "error"]
    assert len(errors) == 1 and errors[0].startswith("Broken failed on a run_start event") and "disk full" in errors[0]


def test_headless_sessions_log_a_sink_failure(isolated: Path, caplog: pytest.LogCaptureFixture):
    class Broken:
        def handle(self, event: Event) -> None:
            raise RuntimeError("disk full")

    session = make_session(
        isolated, ScriptedProvider([text("fine")]), features=(*FEATURES, Feature("b", sinks=lambda s: [Broken()]))
    )
    with caplog.at_level(logging.WARNING, logger="governor.sinks"):
        assert session.ask("go").ok
    assert [r.getMessage() for r in caplog.records] == ["Broken failed on a run_start event: disk full"]


def test_a_failing_report_of_a_sink_failure_never_ends_the_run(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class Broken:
        def handle(self, event: Event) -> None:
            raise RuntimeError("disk full")

    class ClosedUI(ScriptedIO):  # a surface shutting down: telling the human fails too
        def say(self, text: str, kind: Kind = "plain") -> None:
            if kind == "error":
                raise RuntimeError("Event loop is closed")
            super().say(text, kind)

    features = (*FEATURES, Feature("b", sinks=lambda s: [Broken()]))
    session = make_session(isolated, ScriptedProvider([text("fine")]), ClosedUI(), features=features)
    with caplog.at_level(logging.WARNING, logger="governor.sinks"):
        assert session.ask("go").ok
    assert [r.getMessage() for r in caplog.records] == ["Broken failed on a run_start event: disk full"]


def test_an_events_override_still_puts_its_sinks_behind_the_fan_out(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class Broken:
        def handle(self, event: Event) -> None:
            raise RuntimeError("disk full")

    broken = Broken()
    session = make_session(isolated, ScriptedProvider([text("fine")]), events=EventBus([broken]))
    assert session.runtime.events.sinks == [session.sinks] and session.sinks.find(Broken) is broken
    assert session.ask("go").ok


def test_the_model_picker_sets_any_role(isolated: Path):
    io = ScriptedIO(["subagent:sol", "qa:same"])
    session = make_session(isolated, ScriptedProvider(), io)
    session.submit("/model")
    assert Settings.load().selection().roles == {"main": "astra", "subagent": "sol"}
    session.submit("/model")
    assert "subagent model: sol (gpt-6.1-sol)" in io.said() and "qa model: same as the conversation" in io.said()
    assert io.questions == ["Models on openai", "Models on openai"]


def test_status_fragments(isolated: Path):
    io = ScriptedIO()
    session = make_session(isolated, ScriptedProvider(), io)
    assert session.status() == [("openai · astra · subagent: luna · agent: general", ""), ("$0.0000", "")]
    session.submit("/yolo")
    session.submit("/effort high")
    assert session.status()[1:] == [("effort: high", ""), ("$0.0000", ""), ("yolo", "warning")]
    assert "effort: high" in io.said() and "yolo on" in io.said()


def test_a_quiet_run_is_marked_only_while_it_runs(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: list[bool] = []
    sessions: list[Session] = []

    def answer(request: ModelRequest):
        seen.append(sessions[0].quiet_run)
        return text("data")

    sessions.append(make_session(isolated, ScriptedProvider([answer, answer])))
    session = sessions[0]
    assert session.run(session.spec(), "survey", quiet=True).ok and not session.quiet_run
    assert session.ask("talk").ok
    assert seen == [True, False]


def test_console_printer_summarizes_a_runs_tool_calls():
    from agent_core import Event
    from governor.sinks import ConsolePrinter, summarize_calls

    lines: list[str] = []
    printer = ConsolePrinter(write=lines.append)
    calls = [("list_files", {"pattern": "*"}), ("read_file", {"path": "a.py"}), ("read_file", {"path": "b.py"})]
    for i, (name, args) in enumerate(calls):
        printer.handle(Event("tool_call", "r1", "general", data={"id": str(i), "name": name, "input": args}))
        result = {"id": str(i), "name": name, "content": "ok", "is_error": False}
        printer.handle(Event("tool_result", "r1", "general", data=result))
    assert lines == []  # nothing per call unless verbose
    printer.handle(Event("run_end", "r1", "general", data={"stop": "end_turn", "text": "", "usage": {}, "turns": 2}))
    assert len(lines) == 1 and "3 tool calls: 1 list_files, 2 read_file · read a.py, b.py" in lines[0]
    assert summarize_calls([]) == "no tool calls"
    many = [("read_file", {"path": f"f{i}.py"}) for i in range(8)] + [("edit_file", {"path": "f0.py"})]
    expected = "9 tool calls: 8 read_file, 1 edit_file · read f0.py, f1.py, f2.py, f3.py, f4.py, f5.py, +2 more · changed f0.py"
    assert summarize_calls(many) == expected
