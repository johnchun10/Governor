"""The terminal UI, which is what ``govern`` opens: a full-screen Textual app.

Layout, top to bottom: the conversation (scrolls, stays pinned to the newest
output), slash-command suggestions while you type ``/``, the input box, and a
status bar (provider, models, agent, cost, and what is running).

The app is a surface over ``Session`` (``session.py``): it owns no conversation
state and runs no agent itself. Typed lines go to ``Session.submit``; run events
come back through sinks (tool lines through ``ConsolePrinter``, the top-level
reply streamed into one Markdown widget by ``LogSink``); and everything the
application layer asks of the human (pickers, permission and sign-in prompts, a
mode's panel) comes through the ``Interaction`` methods the app implements.
Only ``verbose``, ``help`` and ``exit`` are defined here, because they are
about this screen.

Threading: Textual owns the main thread. Tasks and commands run in worker
threads, so they can block on a picker or a permission prompt (``_modal``
shows a dialog and waits for the answer) while the screen stays live. Output
from workers, including parallel subagents, is mounted through
``call_from_thread``.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from agent_core import Event

from .commands import EFFORT_LEVELS, Command, set_effort
from .config import home
from .models import short_name
from .modes import Mode
from .ports import Kind, Tab
from .runtime import build_session
from .sinks import ConsolePrinter

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
HINT = "Type a task, or / for commands. Ctrl-J adds a line, Esc stops a task, Ctrl-D quits."
PLACEHOLDER = "Ask anything, or type / for commands"
KEYS = (
    "Enter sends · Ctrl-J new line · ↑↓ previous inputs · Tab completes a command\n"
    "Esc or Ctrl-C stops a running task (the model call is cancelled) · Ctrl-D quits · "
    "hold Shift (Option on macOS) while dragging to select\n"
    "Shift-Tab cycles the effort · Ctrl-1…4 set it low…max and Ctrl-0 the default, "
    "in terminals that report Ctrl+digit (kitty keyboard protocol)"
)


class LiveReply:
    """The top-level agent's reply as it streams: one Markdown widget, updated in place.

    ``add`` runs on the worker thread; renders go through the UI thread at most
    every ``interval`` seconds, and ``finish`` draws the final text.
    """

    interval = 0.08

    def __init__(self, app: GovernorApp, classes: str) -> None:
        self.app, self.classes = app, classes
        self.text = ""
        self.widget: Markdown | None = None
        self._last = 0.0

    def add(self, piece: str) -> None:
        self.text += piece
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            self.app.call_from_thread(self._render, self.text)

    def finish(self, text: str) -> None:
        self.text = text
        self.app.call_from_thread(self._render, text)

    async def _render(self, text: str) -> None:
        if self.widget is None:
            self.widget = Markdown(text, classes=self.classes)
            await self.app.target().mount(self.widget)
        else:
            await self.widget.update(text)


class LogSink:
    """The app's sink: the top-level agent's reply streams into a ``LiveReply`` as its deltas arrive.

    Tool and subagent lines reach the log through ``ConsolePrinter(write=...)``;
    this one only follows ``run_start`` / ``delta`` / ``run_end`` at depth 0.
    """

    def __init__(self, app: GovernorApp) -> None:
        self.app = app

    def handle(self, event: Event) -> None:
        if event.depth != 0:
            return
        app = self.app
        if event.type == "run_start":
            app.live = None if app.session.quiet_run else LiveReply(app, "answer")
        elif event.type == "delta" and event.data["kind"] == "text" and app.live is not None:
            app.live.add(event.data["text"])
        elif event.type == "run_end" and app.live is not None:
            live, app.live = app.live, None
            if event.data["stop"] in ("cancelled", "error"):
                if live.widget is not None:
                    live.finish(live.text)  # leave what arrived; the status line says it was stopped
            else:
                live.finish(event.data["text"] or "(no text)")


# --- dialogs -------------------------------------------------------------------------


class ChoiceScreen(ModalScreen[Any]):
    """Arrow-key picker. Dismisses with the chosen value, or None on Esc.

    With ``tabs``, ←/→ switch between pages (the /model roles) and the result is ``(tab id, value)``.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("left", "tab(-1)", "Previous page", show=False),
        Binding("right", "tab(1)", "Next page", show=False),
    ]

    def __init__(
        self,
        title: str,
        options: list[tuple[str, str]] | None = None,
        default: str | None = None,
        *,
        tabs: Sequence[Tab] = (),
        start: str | None = None,
    ) -> None:
        super().__init__()
        self.heading, self.tabs = title, list(tabs)
        self.tab = next((i for i, t in enumerate(self.tabs) if t.id == start), 0)
        if self.tabs:
            self.options, self.default = self.tabs[self.tab].options, self.tabs[self.tab].default
        else:
            self.options, self.default = list(options or []), default

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(Text(self.heading), classes="dialog-title")
            if self.tabs:
                yield Static(self._strip(), classes="dialog-tabs")
            yield OptionList()
            hint = ("←→ switch page · " if self.tabs else "") + "↑↓ move · Enter choose · Esc cancel"
            yield Static(hint, classes="dialog-hint")

    def _strip(self) -> Text:
        strip = Text()
        for i, tab in enumerate(self.tabs):
            strip.append(f" {tab.label} ", style="bold reverse" if i == self.tab else "dim")
            strip.append("  ")
        return strip

    def _fill(self) -> None:
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options([Option(Text(label), id=f"opt{i}") for i, (_, label) in enumerate(self.options)])
        values = [value for value, _ in self.options]
        options.highlighted = values.index(self.default) if self.default in values else 0

    def on_mount(self) -> None:
        self._fill()
        self.query_one(OptionList).focus()

    def action_tab(self, step: int) -> None:
        if not self.tabs:
            return
        self.tab = (self.tab + step) % len(self.tabs)
        self.options, self.default = self.tabs[self.tab].options, self.tabs[self.tab].default
        self.query_one(".dialog-tabs", Static).update(self._strip())
        self._fill()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        value = self.options[event.option_index][0]
        self.dismiss((self.tabs[self.tab].id, value) if self.tabs else value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextScreen(ModalScreen[str]):
    """One-line text (or hidden secret) entry. Dismisses with the text, or "" on Esc."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self, title: str, default: str = "", secret: bool = False, suggestions: list[str] | None = None
    ) -> None:
        super().__init__()
        self.heading, self.default, self.secret, self.suggestions = title, default, secret, suggestions or []

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(Text(self.heading), classes="dialog-title")
            yield Input(
                value=self.default,
                password=self.secret,
                suggester=SuggestFromList(self.suggestions) if self.suggestions else None,
            )
            hint = "Enter accept · Esc cancel" + (" · → completes" if self.suggestions else "")
            yield Static(hint, classes="dialog-hint")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss("")


# --- input box -----------------------------------------------------------------------


class PromptArea(TextArea):
    """Multi-line input: Enter sends, Ctrl-J (or Shift/Alt-Enter where the terminal reports it) adds a line.

    Keys the app wants first (suggestions, input history) go through ``intercept``.
    """

    class Submitted(TextualMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, intercept: Callable[[str], bool], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.intercept = intercept

    async def _on_key(self, event: events.Key) -> None:
        if self.intercept(event.key):
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in ("ctrl+j", "shift+enter", "alt+enter"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


# --- the app -------------------------------------------------------------------------


class GovernorApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #log { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    .banner { color: $text-muted; margin-bottom: 1; }
    .user { background: $boost; padding: 0 1; margin-top: 1; text-style: bold; }
    .dim { color: $text-muted; }
    .error { color: $error; }
    .answer { margin: 1 0 0 0; padding: 0; }
    .mode { height: auto; margin-top: 1; padding-left: 1; border-left: tall $accent; }
    .mode-head { color: $accent; text-style: bold; }
    .mode .user { background: transparent; padding: 0; }
    .mode .answer { margin: 0; }
    #suggest { height: auto; max-height: 10; display: none; margin: 0 1; border: none; background: $surface; }
    #prompt { height: auto; min-height: 3; max-height: 12; margin: 0 1; border: round $accent; }
    #status { height: 1; padding: 0 2; color: $text-muted; }
    ChoiceScreen, TextScreen { align: center middle; }
    .dialog { width: 100; max-width: 95%; height: auto; max-height: 85%; border: round $accent;
              background: $panel; padding: 1 2; }
    .dialog OptionList { height: auto; max-height: 20; border: none; }
    .dialog-title { text-style: bold; margin-bottom: 1; }
    .dialog-tabs { margin-bottom: 1; }
    .dialog-hint { color: $text-muted; margin-top: 1; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt", "Stop", priority=True, show=False),
        Binding("ctrl+d", "quit_if_empty", "Quit", priority=True, show=False),
        Binding("escape", "escape", "Stop", show=False),
        Binding("shift+tab", "cycle_effort", "Effort", priority=True, show=False),
        # Ctrl+digit reaches the app only from terminals with the kitty keyboard protocol.
        *(
            Binding(f"ctrl+{i}", f"effort('{level}')", "Effort", priority=True, show=False)
            for i, level in enumerate(EFFORT_LEVELS)
        ),
    ]
    ENABLE_COMMAND_PALETTE = False
    TITLE = "Governor"

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.printer = ConsolePrinter(write=self._event_line)
        self.session = build_session(project_dir, io=self, sinks=[self.printer, LogSink(self)], trace_label="tui")
        for command in (
            Command("verbose", "toggle showing tool output and all model text", lambda _s, _a: self._verbose()),
            Command("help", "list commands and keys", lambda _s, _a: self._help()),
            Command("exit", "quit (or Ctrl-D)", lambda _s, _a: self.call_from_thread(self.exit)),
        ):
            self.session.commands[command.name] = command
        self.live: LiveReply | None = None
        """The reply being streamed right now, if any (set by ``LogSink``)."""
        self.mode_box: Vertical | None = None
        """While a mode is on, the panel in the log that all output goes to."""
        self.busy: str | None = None
        """What is running ("working", "/login"...), or None when idle."""
        self.busy_since = 0.0
        self._ui_thread = threading.get_ident()
        self._inputs: list[str] = self._load_inputs()
        self._input_index: int | None = None

    # --- layout ------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        # Kept as attributes: query_one() searches the active screen, which is a dialog while one is open.
        self.log_view = VerticalScroll(id="log")
        self.suggest = OptionList(id="suggest")
        self.prompt = PromptArea(self._intercept_key, id="prompt", placeholder=PLACEHOLDER)
        self.status_bar = Static(id="status")
        yield from (self.log_view, self.suggest, self.prompt, self.status_bar)

    def on_mount(self) -> None:
        self._ui_thread = threading.get_ident()
        log = self.log_view
        log.anchor()
        log.mount(Static(Text("Governor  ", style="bold") + Text(str(self.session.project_dir)), classes="banner"))
        self.prompt.focus()
        self.set_interval(0.2, self._refresh_status)
        self._refresh_status()
        if self.session.settings.provider is None:
            self._start("setup", self._first_run)
        else:
            self.say(HINT, "dim")

    def _first_run(self) -> None:
        self.say("Welcome. First, pick where models run (change it any time with /provider).")
        self.session.commands["provider"].run(self.session, "")
        self.say(HINT, "dim")

    # --- output (safe from any thread) ---------------------------------------------------

    def target(self) -> Widget:
        """Where output goes: the mode's panel while one is on, else the log."""
        return self.mode_box if self.mode_box is not None else self.log_view

    def _on_ui(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn`` on the UI thread and wait for it."""
        if threading.get_ident() == self._ui_thread:
            return fn(*args)
        return self.call_from_thread(fn, *args)

    def _post(self, make: Callable[[], Widget]) -> None:
        self._on_ui(lambda: self.target().mount(make()))

    def say(self, text: str, kind: Kind = "plain") -> None:
        if kind == "answer":
            self._post(lambda: Markdown(text, classes="answer"))
        else:
            classes = {"plain": "", "dim": "dim", "error": "error", "user": "user"}[kind]
            self._post(lambda: Static(Text(text), classes=classes))

    def _event_line(self, line: str) -> None:
        self._post(lambda: Static(Text.from_ansi(line)))

    def clear(self) -> None:
        self._on_ui(self.log_view.remove_children)

    def show_mode(self, mode: Mode | None) -> None:
        if mode is None:
            self._on_ui(self._close_mode)
            return
        if threading.get_ident() == self._ui_thread:
            raise RuntimeError("a mode is entered from a worker thread")
        self.call_from_thread(self._open_mode, mode)  # waits until the panel is mounted

    async def _open_mode(self, mode: Mode) -> None:
        heading, *notes = mode.hint.splitlines()
        box = Vertical(
            Static(Text(heading), classes="mode-head"), *(Static(Text(n), classes="dim") for n in notes), classes="mode"
        )
        await self.log_view.mount(box)
        self.mode_box = box
        self.prompt.placeholder = mode.placeholder
        self._refresh_title()

    def _close_mode(self) -> None:
        if self.mode_box is not None:
            self.mode_box.remove()
            self.mode_box = None
        self.prompt.placeholder = PLACEHOLDER
        self._refresh_title()

    def _refresh_title(self) -> None:
        """The input box's title is the mode's to set (Q&A: new question or follow-up)."""
        mode = self.session.mode
        self.prompt.border_title = mode.title(self.prompt.text) if mode is not None else None

    def _refresh_status(self) -> None:
        line = Text()
        for text, style in self.session.status():
            line.append((" · " if line else "") + text, style={"bold": "bold", "warning": "bold yellow"}.get(style, ""))
        if self.busy:
            elapsed = time.monotonic() - self.busy_since
            frame = SPINNER[int(elapsed * 8) % len(SPINNER)]
            running = self.session.running
            label, hint = self.busy, ""
            if running is not None:
                label = short_name(running)
                hint = " · stopping" if self.session.stop.is_set() else " · Esc to stop"
            line.append(f"   {frame} {label} {elapsed:.0f}s{hint}", style="bold")
        self.status_bar.update(line)
        self._refresh_title()

    # --- blocking prompts for worker threads (the Interaction port) -------------------------

    def _modal(self, screen: ModalScreen[Any]) -> Any:
        if threading.get_ident() == self._ui_thread:
            raise RuntimeError("dialogs must be opened from a worker thread")
        done, box = threading.Event(), []

        def show() -> None:
            self.push_screen(screen, callback=lambda value: (box.append(value), done.set()))

        self.call_from_thread(show)
        done.wait()
        return box[0]

    def text(self, message: str, default: str = "", *, suggestions: list[str] | None = None) -> str:
        return self._modal(TextScreen(message, default, suggestions=suggestions)) or ""

    def secret(self, message: str) -> str:
        return self._modal(TextScreen(message, secret=True)) or ""

    def choose(self, message: str, options: list[tuple[str, str]], default: str | None = None) -> str | None:
        return self._modal(ChoiceScreen(message, options, default))

    def choose_tabbed(self, message: str, tabs: Sequence[Tab], start: str) -> tuple[str, str] | None:
        return self._modal(ChoiceScreen(message, tabs=tabs, start=start))

    def confirm(self, message: str, *, yes: str = "Yes", no: str = "No") -> bool:
        return self._modal(ChoiceScreen(message, [("yes", yes), ("no", no)])) == "yes"

    def run_interactive(self, command: list[str]) -> int:
        """Hand the terminal to an interactive command (``aws login``) while the UI is suspended."""

        def suspended() -> int:
            try:
                with self.suspend():
                    return subprocess.run(command, check=False).returncode
            except Exception as exc:  # e.g. no real terminal to suspend to
                self.say(f"couldn't run {command[0]}: {exc}", "error")
                return 1

        return self.call_from_thread(suspended)

    # --- input -------------------------------------------------------------------------

    def _commands(self) -> dict[str, Command]:
        """The commands that apply right now: the session's, with the mode's on top while one is on."""
        mode = self.session.mode
        return {**self.session.commands, **(mode.commands if mode is not None else {})}

    def _intercept_key(self, key: str) -> bool:
        """Keys the input box hands to the app first. True means handled."""
        suggest = self.suggest
        prompt = self.prompt
        if suggest.display and suggest.option_count:
            if key in ("up", "down"):
                current = suggest.highlighted or 0
                suggest.highlighted = (current + (1 if key == "down" else -1)) % suggest.option_count
                return True
            if key in ("tab", "enter"):
                completion = str(suggest.get_option_at_index(suggest.highlighted or 0).id)
                suggest.display = False
                if key == "enter":
                    self._submit(completion)
                else:
                    prompt.text = completion + " "
                    prompt.move_cursor(prompt.document.end)
                return True
        if key in ("up", "down") and self._inputs:
            row = prompt.cursor_location[0]
            last = prompt.document.line_count - 1
            if (key == "up" and row == 0) or (key == "down" and row == last and self._input_index is not None):
                self._recall(-1 if key == "up" else 1)
                return True
        return False

    def _recall(self, step: int) -> None:
        prompt = self.prompt
        index = len(self._inputs) if self._input_index is None else self._input_index
        index = max(0, min(len(self._inputs), index + step))
        self._input_index = None if index == len(self._inputs) else index
        prompt.text = "" if self._input_index is None else self._inputs[index]
        prompt.move_cursor(prompt.document.end)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        suggest = self.suggest
        options: list[Option] = []
        if text.startswith("/") and "\n" not in text:
            commands = self._commands()
            name, space, arg = text[1:].partition(" ")
            if not space:
                options = [
                    Option(Text(f"/{n:<10}", style="bold") + Text(c.help, style="dim"), id=f"/{n}")
                    for n, c in commands.items()
                    if n.startswith(name)
                ]
            elif (command := commands.get(name)) and command.args:
                completions = command.args(self.session)
                options = [Option(f"/{name} {a}", id=f"/{name} {a}") for a in completions if a.startswith(arg)]
            if len(options) == 1 and options[0].id == text.strip():
                options = []  # already complete
        suggest.set_options(options)
        suggest.display = bool(options)
        if options:
            suggest.highlighted = 0
        self._refresh_title()

    def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        self._submit(event.text)

    def _submit(self, raw: str) -> None:
        line = raw.strip()
        if not line:
            return
        if self.busy:
            self.notify(f"{self.busy} is still running; wait, or press Esc to stop it", severity="warning")
            return
        self.prompt.text = ""
        self.suggest.display = False
        self._remember_input(line)
        label = line.partition(" ")[0] if line.startswith("/") else "working"
        self._start(label, lambda: self.session.submit(line))

    def _start(self, label: str, work: Callable[[], None]) -> None:
        self.busy, self.busy_since = label, time.monotonic()

        def body() -> None:
            try:
                work()
            except Exception as exc:  # a bug in a command shouldn't kill the UI
                self.say(f"{type(exc).__name__}: {exc}", "error")
            finally:
                self.busy = None

        self.run_worker(body, thread=True, exit_on_error=False)

    # --- history of typed inputs ----------------------------------------------------------

    def _load_inputs(self) -> list[str]:
        try:
            lines = (home() / "inputs.jsonl").read_text().splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines[-500:] if line.strip()]

    def _remember_input(self, line: str) -> None:
        self._input_index = None
        if self._inputs and self._inputs[-1] == line:
            return
        self._inputs.append(line)
        try:
            home().mkdir(parents=True, exist_ok=True)
            with (home() / "inputs.jsonl").open("a") as fh:
                fh.write(json.dumps(line) + "\n")
        except OSError:
            pass

    # --- keys ----------------------------------------------------------------------------

    def action_interrupt(self) -> None:
        if self.session.running is not None:
            self.session.request_stop()
            return
        prompt = self.prompt
        if prompt.text:
            prompt.text = ""
        else:
            self.notify("Ctrl-D quits")

    def action_escape(self) -> None:
        suggest = self.suggest
        if suggest.display:
            suggest.display = False
        elif self.session.running is not None:
            self.session.request_stop()
        elif self.session.mode is not None and not self.busy and not self.prompt.text:
            self.session.leave_mode()

    def action_quit_if_empty(self) -> None:
        if not isinstance(self.screen, ModalScreen) and not self.prompt.text:
            self.exit()
        elif self.focused is self.prompt:
            self.prompt.action_delete_right()  # what Ctrl-D does in the input box otherwise

    def action_effort(self, level: str) -> None:
        self._set_effort(level)

    def action_cycle_effort(self) -> None:
        current = self.session.settings.effort or EFFORT_LEVELS[0]
        self._set_effort(EFFORT_LEVELS[(EFFORT_LEVELS.index(current) + 1) % len(EFFORT_LEVELS)])

    def _set_effort(self, level: str) -> None:
        """A key sets the level; a toast says so instead of a log line."""
        self.notify(set_effort(self.session, level), timeout=2)
        self._refresh_status()

    # --- this screen's own commands ---------------------------------------------------------

    def _verbose(self) -> None:
        self.printer.verbose = not self.printer.verbose
        self.say("verbose on: tool output and all model text are shown" if self.printer.verbose else "verbose off")

    def _help(self) -> None:
        for name, command in self.session.commands.items():
            self.say(f"/{name:<10} {command.help}")
        mode = self.session.mode
        if mode is not None:
            for name, command in mode.commands.items():
                self.say(f"/{name:<10} {command.help} (in {mode.name})")
            self.say(mode.hint, "dim")
        self.say(KEYS, "dim")
