"""The living conversation and the one run sequence, with no UI in it.

``Session`` owns what the surfaces used to keep as widget state: the history,
the (profile, model) that produced it, the saved chat it belongs to, the cost
total, the trace, the stop event and the active input mode. ``run()`` is the only
place besides core's ``task`` tool that constructs an ``Agent``: the sign-in
offer, the trace, the cost total and the error handling happen here, once, and
``ask()``, Q&A, scripted runs and future tickets all go through it. Results come
back as data (``Turn``, ``Resumed``, status fragments); the surface decides how
to show them. Questions to the human go through ``io`` (``ports.Interaction``);
with ``io=None`` the session never asks, and never refuses a run for lack of a
prompter.

Threading: a run blocks the thread that calls it, and so do ``io`` prompts. The
session never checks which thread it is on and never calls a surface except
through ``io`` and the run's sinks; a surface marshals to its own thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agent_core import (
    Agent,
    AgentSpec,
    Message,
    ModelInfo,
    ProviderError,
    ReasoningBlock,
    RunResult,
    Runtime,
    ToolResultBlock,
    Usage,
)

from . import auth
from .chats import Chat, ChatStore
from .config import Settings
from .doctor import ping
from .models import resolve, short_name
from .sinks import FanOut, JsonlTrace

if TYPE_CHECKING:
    from .commands import Command
    from .modes import Mode
    from .ports import Interaction, Kind


def summary(usage: Usage, seconds: float | None = None) -> str:
    cache = (
        f", cache r/w {usage.cache_read_tokens}/{usage.cache_write_tokens}"
        if usage.cache_read_tokens or usage.cache_write_tokens
        else ""
    )
    took = f" · {seconds:.1f}s" if seconds is not None else ""
    return f"{usage.input_tokens} in / {usage.output_tokens} out tokens{cache} · ${usage.cost_usd:.4f}{took}"


def strip_reasoning(messages: Sequence[Message]) -> list[Message]:
    """``messages`` without their reasoning blocks, for reuse under another model: it can't read them,
    may reject them, and can't use the prompt cache they belong to anyway."""
    return [Message(m.role, tuple(b for b in m.content if not isinstance(b, ReasoningBlock))) for m in messages]


@dataclass(frozen=True)
class Turn:
    """What one run produced, as data.

    ``ran`` is False when the run never started: ``error`` then says why, or is None
    when the user declined (a sign-in offer). ``error`` is also set when the model
    call failed mid-run (``result.stop == "error"``).
    """

    result: RunResult | None = None
    model: ModelInfo | None = None
    seconds: float = 0.0
    error: str | None = None
    stopped: str | None = None
    """What stopped the run, leaving the conversation as it was: ``"user"`` (Esc), or the name of the
    feature that asked (``"budget"``, see ``Session.request_stop``). None when the run ended on its own."""

    @property
    def ran(self) -> bool:
        return self.result is not None

    @property
    def ok(self) -> bool:
        return self.ran and not self.stopped and self.error is None

    @property
    def text(self) -> str:
        return self.result.text if self.result is not None else ""

    @property
    def usage(self) -> Usage:
        return self.result.usage if self.result is not None else Usage()

    def status(self) -> str:
        """The one-line report: model, tokens, cost, time, and how the run stopped if not by itself."""
        if self.result is None or self.model is None:
            return ""
        reason = self.stopped if self.stopped and self.stopped != "user" else self.result.stop
        stop = "" if reason == "end_turn" else f" · stopped: {reason}"
        return f"[{short_name(self.model)}] {summary(self.result.usage, self.seconds)}{stop}"


@dataclass(frozen=True)
class Resumed:
    """A saved chat made the conversation again."""

    chat: Chat
    messages: list[Message]
    switched: bool
    """The chat ran on another provider: its reasoning was dropped and the current main model continues it."""
    note: str


class Session:
    """One conversation and the runs that extend it. Built by ``runtime.build_session``."""

    def __init__(
        self,
        project_dir: Path,
        settings: Settings,
        runtime: Runtime,
        io: Interaction | None,
        commands: Sequence[Command],
        chats: ChatStore | None = None,
        *,
        sinks: FanOut | None = None,
        status: Sequence[Callable[[Session], Sequence[tuple[str, str]]]] = (),
        trace_label: str = "session",
        closers: Sequence[Callable[[], None]] = (),
    ) -> None:
        self.project_dir = project_dir
        self.settings = settings
        self.runtime = runtime
        self.io = io
        self.commands: dict[str, Command] = {c.name: c for c in commands}
        self.chats = chats
        """Where completed turns are saved; None keeps the session chat-less."""
        if runtime.stop is None:
            runtime.stop = threading.Event()
        self.stop: threading.Event = runtime.stop
        self.sinks: FanOut = sinks if sinks is not None else FanOut()
        """Where every run's events go. Register a sink here, never on the bus: one that raises is skipped, not fatal."""
        if self.sinks not in runtime.events.sinks:
            runtime.events.sinks.append(self.sinks)
        self.history: list[Message] | None = None
        self.conversation: tuple[AgentSpec, ModelInfo] | None = None
        """The profile and model that produced ``history``. Q&A answers with them, so it reads the same cache."""
        self.chat: Chat | None = None
        """The saved chat that ``history`` belongs to, once a turn has completed."""
        self.total = Usage()
        self.trace: Path | None = None
        self.mode: Mode | None = None
        self.running: ModelInfo | None = None
        """The model of the run in progress, or None."""
        self.quiet_run = False
        """The run in progress produces data, not a reply (a survey's JSON): a surface shows no answer for it."""
        self._trace_label = trace_label
        self._trace_sink: JsonlTrace | None = None
        self._status = list(status)
        """Each feature's status-bar fragments, after the harness's own."""
        self._stopped_by: str | None = None
        self._closers = list(closers)

    # --- through the human port (no-ops when headless) ---------------------------------

    def say(self, text: str, kind: Kind = "plain") -> None:
        if self.io is not None:
            self.io.say(text, kind)

    def choose(self, message: str, options: list[tuple[str, str]], default: str | None = None) -> str | None:
        return self.io.choose(message, options, default) if self.io is not None else None

    def input(self, message: str, default: str = "", *, suggestions: list[str] | None = None) -> str:
        return self.io.text(message, default, suggestions=suggestions) if self.io is not None else default

    # --- sign-in ---------------------------------------------------------------------------

    def offer_sign_in(self, provider: str) -> bool:
        """Ask whether to sign in to ``provider`` now, and do it. False when headless or declined."""
        if self.io is None:
            return False
        question = f"You're not signed in to {provider}. Sign in now?"
        return self.io.confirm(question, yes="Sign in", no="Not now") and self.sign_in(provider)

    def sign_in(self, provider: str) -> bool:
        """Run the provider's login flow, then check the active provider with a tiny request."""
        if self.io is None or not auth.login(provider, self.io):
            return False
        sel = self.settings.selection()
        if provider != sel.provider:
            return True
        probe = resolve("subagent", sel, self.project_dir)
        self.say(f"checking with a tiny request to {short_name(probe)}…", "dim")
        ok, detail = ping(probe)
        self.say(f"{'✓' if ok else '✗'} {detail}", "plain" if ok else "error")
        return ok

    # --- runs ------------------------------------------------------------------------------

    def spec(self) -> AgentSpec:
        """The agent profile the settings name (``general`` when it doesn't exist)."""
        profiles = self.runtime.profiles
        return profiles.get(self.settings.agent) or profiles["general"]

    def run(
        self,
        spec: AgentSpec,
        prompt: str,
        *,
        history: Sequence[Message] | None = None,
        runtime: Runtime | None = None,
        quiet: bool = False,
    ) -> Turn:
        """Run ``spec`` on ``prompt`` after ``history``: the one run sequence.

        ``runtime`` is a per-run variation of the session's (``replace(session.runtime, ...)``),
        sharing its event bus and stop event. ``quiet`` marks a run whose final text is data for the
        caller (a survey's JSON), so a surface shows its tool lines but not an answer. Sign-in is
        offered only when ``io`` is present; otherwise the run proceeds and the provider reports.
        Nothing here commits to the conversation: ``ask`` does that, Q&A never does.
        """
        rt = runtime or self.runtime
        try:
            model = rt.resolve_model(rt.model_override or spec.model)
        except (KeyError, ValueError) as exc:
            return Turn(error=str(exc).strip("'\""))
        if self.io is not None and not auth.status(model.provider).ok and not self.offer_sign_in(model.provider):
            return Turn()
        try:
            agent = Agent(spec, rt, cwd=self.project_dir)
        except (KeyError, ValueError, ProviderError) as exc:
            return Turn(error=str(exc).strip("'\""))
        self._open_trace()
        self.stop.clear()
        self._stopped_by = None
        self.running, self.quiet_run = agent.model, quiet
        started = time.monotonic()
        try:
            result = agent.run(prompt, history=list(history or []))
        finally:
            self.running, self.quiet_run = None, False
        self.total = self.total + result.usage
        # A feature's stop counts only when the loop cut something short: it cancelled a call, or refused the
        # tool calls of the last turn it had (max_turns). A cap crossed by the run's final reply, however that
        # reply ended (end_turn, max_tokens, content_filtered), leaves a finished, paid-for turn, which is kept.
        # The user's Esc counts even when the reply beat it, as before.
        by = self._stopped_by or "user"
        refused_last_tools = (
            result.stop == "max_turns"
            and self.stop.is_set()
            and bool(result.messages)
            and any(isinstance(b, ToolResultBlock) for b in result.messages[-1].content)
        )
        cut_short = result.stop == "cancelled" or (refused_last_tools and by != "user")
        stopped = cut_short or (self.stop.is_set() and by == "user")
        return Turn(
            result,
            agent.model,
            time.monotonic() - started,
            error=result.text if result.stop == "error" else None,
            stopped=by if stopped else None,
        )

    def ask(self, task: str) -> Turn:
        """Continue the conversation with ``task`` under the current agent; a completed turn is kept and saved."""
        spec = self.spec()
        turn = self.run(spec, task, history=self.history)
        if turn.ok and turn.result is not None and turn.model is not None:
            self.history = turn.result.messages
            self.conversation = (spec, turn.model)
            self._save_chat(task, turn.result, spec, turn.model)
        return turn

    def request_stop(self, by: str = "user") -> None:
        """Stop the run in progress: the model call is cancelled and pending tool calls are refused.

        ``by`` names who asked and is reported as ``Turn.stopped``: the user (the default, which also
        says so), or a feature's name when its sink or command stops a run (it says why itself).
        """
        if self.stop.is_set():
            if by == "user" and self._stopped_by != "user":  # Esc after a feature's stop: the user's stop wins
                self._stopped_by = "user"
                self.say("stopping: the model call is cancelled and pending tool calls are refused", "dim")
            return
        self._stopped_by = by
        self.stop.set()
        if by == "user":
            self.say("stopping: the model call is cancelled and pending tool calls are refused", "dim")

    def _open_trace(self) -> None:
        if self.trace is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.trace = self.project_dir / ".governor" / "runs" / f"{stamp}-{self._trace_label}.jsonl"
            self._trace_sink = JsonlTrace(self.trace)
            self.sinks.add(self._trace_sink)

    def close(self) -> None:
        """Release what the session and its features hold open. Final; the trace path stays readable."""
        if self._trace_sink is not None:
            self.sinks.remove(self._trace_sink)
            self._trace_sink.close()
            self._trace_sink = None
        for close in self._closers:
            close()

    # --- the conversation record -----------------------------------------------------------

    def _save_chat(self, task: str, result: RunResult, spec: AgentSpec, model: ModelInfo) -> None:
        if self.chats is None:
            return
        if self.chat is None:
            self.chat = Chat.new(task, self.settings.selection().provider, model.alias, spec.name)
        self.chat.messages = list(result.messages)
        self.chat.usage = self.chat.usage + result.usage
        self.chat.model, self.chat.agent = model.alias, spec.name
        try:
            self.chats.save(self.chat)
        except OSError as exc:
            self.say(f"couldn't save the chat: {exc}", "dim")

    def resume(self, chat: Chat) -> Resumed:
        """Make ``chat`` the conversation. Any mode is left; the surface shows the transcript."""
        sel = self.settings.selection()
        spec = self.runtime.profiles.get(chat.agent) or self.spec()
        try:
            model = resolve(chat.model, sel, self.project_dir)
        except KeyError:
            model = resolve("main", sel, self.project_dir)
        messages = list(chat.messages)
        switched = model.provider != sel.provider
        if switched:
            # Another provider can't use this chat's prompt cache and may reject its reasoning blocks.
            model = resolve("main", sel, self.project_dir)
            messages = strip_reasoning(messages)
        self.leave_mode()
        self.chat, self.history, self.conversation = chat, messages, (spec, model)
        note = f"{chat.turns} turns · agent {chat.agent} · {chat.model}"
        if switched:
            note += f" (now on {sel.provider}: its prompt cache and reasoning don't carry over)"
        return Resumed(chat, messages, switched, note)

    def clear(self) -> None:
        """Start a new conversation: forget the history and its chat, and leave any mode."""
        self.leave_mode()
        self.history, self.conversation, self.chat = None, None, None

    # --- modes and typed input ---------------------------------------------------------------

    def enter_mode(self, mode: Mode) -> None:
        if self.mode is not None:
            self.leave_mode()
        self.mode = mode
        if self.io is not None:
            self.io.show_mode(mode)

    def leave_mode(self) -> None:
        mode, self.mode = self.mode, None
        if mode is None:
            return
        mode.leave(self)
        if self.io is not None:
            self.io.show_mode(None)

    def submit(self, line: str) -> None:
        """Typed input: a slash command (the mode's, then the session's), the mode's own input, or a task."""
        line = line.strip()
        if not line:
            return
        if line.startswith("/"):
            name, _, arg = line[1:].partition(" ")
            command = (self.mode.commands.get(name) if self.mode is not None else None) or self.commands.get(name)
            if command is None or command.echo:
                self.say("› " + line, "user")
            if command is None:
                self.say(f"unknown command /{name}; try /help", "error")
                return
            command.run(self, arg.strip())
        elif self.mode is not None:
            self.mode.submit(self, line)
        else:
            self.say("› " + line, "user")
            self._report(self.ask(line))

    def _report(self, turn: Turn) -> None:
        if turn.error is not None:
            self.say(turn.error, "error")
            if turn.ran:
                self.say("the conversation is unchanged; /doctor checks access, /login signs in again", "dim")
        elif turn.stopped == "user":
            self.say("stopped; the conversation is as it was before this message", "dim")
        elif turn.stopped:
            self.say(f"stopped by {turn.stopped}; the conversation is as it was before this message", "dim")
        elif turn.ran:
            self.say(turn.status(), "dim")

    def status(self) -> list[tuple[str, str]]:
        """Status-bar fragments as (text, style), style one of "", "bold" or "warning"."""
        sel = self.settings.selection()
        parts = [
            (
                f"{sel.provider} · {sel.roles['main']} · subagent: {sel.roles['subagent']} · agent: {self.settings.agent}",
                "",
            )
        ]
        if level := self.settings.effort_level():
            parts.append((f"effort: {level}", ""))
        parts.append((f"${self.total.cost_usd:.4f}", ""))
        for fragments in self._status:
            parts.extend(fragments(self))
        if self.mode is not None:
            parts.append((self.mode.name, "bold"))
        if getattr(self.runtime.permissions, "yolo", False):
            parts.append(("yolo", "warning"))
        return parts
