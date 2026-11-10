"""The full-screen TUI, driven headlessly with Textual's pilot (real key presses, scripted model)."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from textual.widget import Widget
from textual.widgets import Markdown, Static

from agent_core import (
    Message,
    ModelRequest,
    ModelResponse,
    ReasoningBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    Usage,
)
from agent_core.testing import ScriptedProvider, text, tool_call
from governor.chats import ChatStore
from governor.config import Settings
from governor.effort import wrap as with_effort
from governor.modes.qa import QAMode
from governor.tui import ChoiceScreen, GovernorApp

pytestmark = pytest.mark.usefixtures("isolated")


def drive(app: GovernorApp, script: Callable[[object], Awaitable[None]]) -> str:
    """Run ``script`` against the app; returns the transcript as it was at the end."""
    captured: list[str] = []

    async def main() -> None:
        async with app.run_test(size=(110, 36)) as pilot:
            await script(pilot)
            captured.append(transcript(app))

    asyncio.run(main())
    return captured[0]


async def until(pilot, condition: Callable[[], bool], timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        await pilot.pause(0.05)
        if condition():
            return
    raise AssertionError("timed out waiting for the UI")


async def send(pilot, app: GovernorApp, line: str) -> None:
    await pilot.click("#prompt")
    await pilot.press(*line, "enter")
    await until(pilot, lambda: not app.busy or isinstance(app.screen, ChoiceScreen))


def transcript(app: GovernorApp) -> str:
    def lines(parent: Widget) -> Iterator[str]:
        for widget in parent.children:
            if isinstance(widget, Markdown):
                yield widget.source
            elif isinstance(widget, Static):
                yield str(widget.render())
            else:
                yield from lines(widget)  # the Q&A box

    return "\n".join(lines(app.log_view))


def make_app(project: Path, provider: ScriptedProvider | None = None, saved: str | None = "openai") -> GovernorApp:
    if saved:
        Settings(provider=saved).save()
    app = GovernorApp(project)
    if provider is not None:
        app.session.runtime.provider_for = with_effort(
            lambda info: provider, app.session.settings.effort_level
        )  # as build_runtime does
    return app


def test_first_run_asks_for_a_provider(isolated: Path):
    app = make_app(isolated, saved=None)

    async def script(pilot) -> None:
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press("down", "down", "down", "enter")  # openai
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen) and "signed in" in app.screen.heading)
        await pilot.press("down", "enter")  # not now
        await until(pilot, lambda: not app.busy)

    log = drive(app, script)
    assert Settings.load().provider == "openai"
    assert "provider: openai · main: astra · subagent: luna" in log


def test_model_commands_and_status_bar(isolated: Path):
    app = make_app(isolated)

    async def script(pilot) -> None:
        await send(pilot, app, "/model sol")
        await send(pilot, app, "/model nope")
        await send(pilot, app, "/agent explore")
        await send(pilot, app, "/nonsense")
        await send(pilot, app, "/model")  # the picker: ←/→ switch between the main, subagent and Q&A pages
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press("right", "up", "enter")  # subagent page; one above the current luna is terra
        await until(pilot, lambda: not app.busy)
        await pilot.pause(0.3)
        status = str(app.status_bar.render())
        assert "openai · sol · subagent: terra · agent: explore" in status

    log = drive(app, script)
    assert "main model: sol (gpt-6.1-sol)" in log and "subagent model: terra (gpt-6.1-terra)" in log
    assert "unknown openai model 'nope'" in log
    assert "unknown command /nonsense" in log
    assert Settings.load().selection().roles["main"] == "sol"


def test_slash_suggestions_complete_and_run(isolated: Path):
    app = make_app(isolated)

    async def script(pilot) -> None:
        await pilot.click("#prompt")
        await pilot.press("/", "m", "o")
        suggest = app.suggest
        assert suggest.display and suggest.option_count == 1
        await pilot.press("tab")
        assert app.prompt.text == "/model "
        await pilot.press("l")
        assert str(suggest.get_option_at_index(0).id) == "/model luna"
        await pilot.press("enter")
        await until(pilot, lambda: not app.busy)

    drive(app, script)
    assert Settings.load().selection().roles["main"] == "luna"


def test_task_asks_permission_and_keeps_the_conversation(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([tool_call("bash", {"command": "echo hi"}), text("**done**"), text("again")])
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await send(pilot, app, "run echo")
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        assert "Allow bash" in app.screen.heading
        await pilot.press("enter")  # Yes
        await until(pilot, lambda: not app.busy)
        await send(pilot, app, "more")
        await until(pilot, lambda: not app.busy)
        await pilot.press("up")
        assert app.prompt.text == "more"

    log = drive(app, script)
    assert "**done**" in log
    assert provider.requests[0].model == "gpt-6-astra"
    assert len(provider.requests[2].messages) == 5  # the first turn was carried over
    assert app.session.trace is not None and app.session.trace.is_file()


def test_ctrl_c_stops_a_running_task(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    gate = threading.Event()

    def slow(request: ModelRequest):
        gate.wait(5)
        return tool_call("bash", {"command": "rm -rf build"})

    provider = ScriptedProvider([slow])
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await pilot.click("#prompt")
        await pilot.press(*"clean up", "enter")
        await until(pilot, lambda: app.busy is not None and len(provider.requests) == 1)
        await pilot.press("ctrl+c")
        assert app.session.stop.is_set()
        gate.set()
        await until(pilot, lambda: not app.busy)

    log = drive(app, script)
    assert "stopped; the conversation is as it was" in log
    assert app.session.history is None
    assert len(provider.requests) == 1  # the call was cancelled; no tool ran and no further model call was made


def test_qa_answers_beside_the_conversation_and_leaves_it_alone(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    replies = ["built it", "**because**", "a second answer", "a follow-up answer", "back to work"]
    provider = ScriptedProvider([text(r) for r in replies])
    app = make_app(isolated, provider)
    during: list[str] = []

    async def script(pilot) -> None:
        await send(pilot, app, "build it")
        await send(pilot, app, "/qa why?")
        assert app.prompt.border_title == "Q&A · new question"
        await send(pilot, app, "what else?")
        await pilot.press("+")
        await pilot.pause(0.1)
        assert app.prompt.border_title == "Q&A · follow-up"
        await pilot.press(*" more?", "enter")
        await until(pilot, lambda: not app.busy)
        during.append(transcript(app))
        await pilot.press("escape")
        await until(pilot, lambda: app.mode_box is None)
        await send(pilot, app, "continue")

    log = drive(app, script)
    main, first, second, follow_up, after = provider.requests
    conversation = (Message.user("build it"), text("built it").message)
    assert main.messages == conversation[:1]
    # Same prompt as the conversation, so the provider can serve it from the prompt cache.
    for qa in (first, second, follow_up):
        assert (qa.model, qa.system, qa.tools, qa.params) == (main.model, main.system, main.tools, main.params)
        assert qa.messages[:2] == conversation
    assert len(first.messages) == 3 and first.messages[2].text.startswith("<qa>")
    assert first.messages[2].text.endswith("why?")
    assert len(second.messages) == 3 and second.messages[2].text.endswith("what else?")  # starts afresh
    assert follow_up.messages[2:] == (second.messages[2], text("a second answer").message, Message.user("more?"))
    assert after.messages == (*conversation, Message.user("continue"))
    assert "? why?" in during[0] and "**because**" in during[0] and "+ more?" in during[0]
    assert "because" not in log and "why?" not in log and "Q&A" not in log
    assert "back to work" in log
    assert app.session.mode is None and app.prompt.border_title is None


def test_qa_refuses_tool_calls_without_asking(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([text("done"), tool_call("bash", {"command": "rm -rf build"}), text("from memory")])
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await send(pilot, app, "do it")
        await send(pilot, app, "/qa what changed?")
        assert not isinstance(app.screen, ChoiceScreen)

    log = drive(app, script)
    refused = provider.requests[2].messages[-1].content[0]
    assert isinstance(refused, ToolResultBlock) and refused.is_error and "tools are off" in refused.content
    assert "from memory" in log
    mode = app.session.mode
    assert isinstance(mode, QAMode) and mode.qa.thread[-1].text == "from memory"


def test_qa_needs_a_conversation(isolated: Path):
    app = make_app(isolated)

    async def script(pilot) -> None:
        await send(pilot, app, "/qa anything?")

    log = drive(app, script)
    assert "there isn't one yet" in log
    assert app.session.mode is None and app.mode_box is None


def test_qa_model_can_differ_from_the_conversation(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    thinking = ReasoningBlock(raw={"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}, provider="openai")
    first = ModelResponse(Message("assistant", (thinking, TextBlock("hi"))), StopReason.END_TURN, Usage(10, 5), "fake")
    provider = ScriptedProvider([first, text("cheap answer")])
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await send(pilot, app, "hello")
        await send(pilot, app, "/qa")
        await send(pilot, app, "/model luna")  # in Q&A, /model is the model that answers
        assert Settings.load().selection().roles == {"main": "astra", "subagent": "luna", "qa": "luna"}
        await send(pilot, app, "why?")
        await send(pilot, app, "/model same")
        await send(pilot, app, "/qa-model luna")

    log = drive(app, script)
    qa = provider.requests[1]
    assert qa.model == "gpt-6-luna"
    assert not any(isinstance(b, ReasoningBlock) for m in qa.messages for b in m.content)
    history = app.session.history
    assert history is not None and history[1].content[0] is thinking  # the conversation keeps it
    assert "luna can't reuse astra's prompt cache" in log
    assert "qa" not in Settings.load().selection().roles
    assert "unknown command /qa-model" in log  # Q&A's commands are its own: /model and /done inside it


def test_signed_out_task_offers_login(isolated: Path):
    provider = ScriptedProvider()
    app = make_app(isolated, provider, saved="anthropic")

    async def script(pilot) -> None:
        await send(pilot, app, "hello")
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press("down", "enter")  # not now
        await until(pilot, lambda: not app.busy)

    drive(app, script)
    assert not provider.requests


def test_ctrl_d_quits_only_when_the_input_is_empty(isolated: Path):
    app = make_app(isolated)

    async def script(pilot) -> None:
        await pilot.click("#prompt")
        await pilot.press("a", "b", "left", "ctrl+d")
        assert app.prompt.text == "a" and app.is_running
        await pilot.press("ctrl+c", "ctrl+d")  # Ctrl-C clears the input
        await pilot.pause(0.2)
        assert not app.is_running

    drive(app, script)


def test_reply_streams_into_the_log(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: list[str] = []
    provider = ScriptedProvider([text("streamed **reply** here")])
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await pilot.click("#prompt")
        await pilot.press(*"go", "enter")
        for _ in range(100):
            await pilot.pause(0.05)
            if app.live is not None and app.live.widget is not None:
                seen.append(app.live.widget.source)
            if not app.busy:
                break

    log = drive(app, script)
    assert "streamed **reply** here" in log
    assert log.count("streamed") == 1  # one widget, updated in place, not a second copy at the end
    assert all("streamed **reply** here".startswith(s) for s in seen)


def test_chats_are_saved_and_resumed(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([text("first answer"), text("second answer"), text("after resume")])
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await send(pilot, app, "first task")
        await send(pilot, app, "/clear")
        await send(pilot, app, "another chat")
        await send(pilot, app, "/chats")
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        assert "Resume a chat" in app.screen.heading
        await pilot.press("down", "enter")  # the older one (newest first)
        await until(
            pilot, lambda: not app.busy and app.session.chat is not None and app.session.chat.title == "first task"
        )
        await send(pilot, app, "continue")

    log = drive(app, script)
    chats = ChatStore(isolated).list()
    assert [c.title for c in chats] == ["first task", "another chat"]
    assert [m.text for m in chats[0].messages] == ["first task", "first answer", "continue", "after resume"]
    assert chats[0].turns == 2 and chats[0].usage.input_tokens == 20
    assert "resumed: first task" in log and "first answer" in log and "another chat" not in log
    assert [m.text for m in provider.requests[2].messages] == ["first task", "first answer", "continue"]
    assert app.session.conversation is not None and app.session.conversation[1].alias == "openai/astra"


def test_chats_without_any_saved(isolated: Path):
    app = make_app(isolated)

    async def script(pilot) -> None:
        await send(pilot, app, "/chats")

    assert "no saved chats" in drive(app, script)


def test_effort_command_and_keys(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([text("one"), text("two"), text("three")])
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await send(pilot, app, "/effort high")
        await send(pilot, app, "first")
        await pilot.press("ctrl+4")  # max
        await pilot.pause(0.3)
        assert "effort: max" in str(app.status_bar.render())
        await send(pilot, app, "second")
        await pilot.press("shift+tab")  # max wraps around to the default
        await pilot.pause(0.3)
        assert "effort" not in str(app.status_bar.render())
        await send(pilot, app, "third")
        await send(pilot, app, "/effort silly")

    log = drive(app, script)
    assert "effort: high" in log and "unknown effort 'silly'" in log
    assert [r.params for r in provider.requests] == [
        {"reasoning": {"effort": "high"}},
        {"reasoning": {"effort": "xhigh"}},
        {},
    ]
    assert Settings.load().effort is None


def test_a_survey_shows_tool_summary_lines_but_no_answer(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    (isolated / "README.md").write_text("# demo\n")
    survey = json.dumps({"areas": [{"path": "README.md", "purpose": "Overview"}], "features": []})
    provider = ScriptedProvider(
        [tool_call("list_files", {"pattern": "**/*"}), tool_call("read_file", {"path": "README.md"}), text(survey)]
    )
    app = make_app(isolated, provider)

    async def script(pilot) -> None:
        await send(pilot, app, "/map survey")
        await until(pilot, lambda: not app.busy)

    log = drive(app, script)
    assert '"areas"' not in log  # the map as JSON is data, not an answer
    assert "2 tool calls: 1 list_files, 1 read_file · read README.md" in log
    assert "⏺ list_files(" not in log  # one summary line, not a line per call
    assert "map: 1 areas · 0 features" in log and app.live is None
