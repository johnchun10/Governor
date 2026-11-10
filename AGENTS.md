# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Governor itself, and others)
working in this repository. Humans should start with `USER_GUIDE.md`; maintainers
should also read `README.md` and `src/agent_core/README.md`.

## What this repo is

Governor is a small, modular agent harness for Amazon Bedrock (plus the Anthropic and OpenAI APIs), written in Python 3.11+. It is
two packages, and the dependency goes one way only: `governor` → `agent_core`.

| Package | Role | Change policy |
| --- | --- | --- |
| `src/agent_core/` | **Primitives**: types, interfaces, the agent loop, event schema. Standard library only. | **Protected.** Never edit in a feature PR. |
| `src/governor/` | **Governor**: providers, per-provider model catalog, settings/login, tools, agent profiles, context providers, permissions, the session with its commands, modes and features, the terminal UI. | Normal feature work happens here. |

## Setup and commands

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest                                   # all tests, offline (no AWS calls)
ruff check . && ruff format --check .    # lint + format check (CI runs both)
ruff format .                            # apply formatting
python tests/test_interfaces.py          # regenerate README's Interfaces section after changing a public class or function
govern                                  # the terminal UI; /doctor makes live checks
```

Run `pytest` and both ruff commands before you consider a change done.

## Where things go

| You want to… | Do this | Touch core? |
| --- | --- | --- |
| Add a tool | Write a `@tool` function in `src/governor/tools/`, register it in `tools/__init__.py:builtin_tools` (or in your feature's `Feature.tools`) | No |
| Add a feature | A module plus one `Feature` entry in `src/governor/features.py`: its tools, context providers, sinks, its one global command, its status-bar fragments (empty unless it is on) and a close hook; a `Mode` if it changes what typed input means. It never edits `tui.py` | No |
| Keep a feature's setting | `session.settings.of("<feature>")`: a dict the feature owns, saved with the rest (`budget.py`). Never a field on `Settings` | No |
| Stop a run from a feature | `session.request_stop("<feature>")` from its sink or command; the turn reports `stopped == "<feature>"` and the session says `stopped by <feature>` (`budget.py`) | No |
| Add a tool in another language | Add a JSON manifest in `.governor/tools/` (see `src/governor/tools/external.py`) | No |
| Add or change an agent | Edit/add `src/governor/agents/<name>.md` (frontmatter + prompt) | No |
| Add or re-point a model | `src/governor/models.py` (`BUILTIN`, `DEFAULT_ROLES`) or `.governor/models.toml` | No |
| Add a model provider | New module in `src/governor/providers/`, register in `providers/__init__.py` (`PROVIDERS`, `get_provider`), add its models and roles in `models.py` and its sign-in in `auth.py` | No |
| Inject knowledge / retrieval / memory | Implement `ContextProvider` (`system`, `before_turn`) in governor or a new package; register it through its `Feature.context_providers` in `features.py` (or `build_session(context_providers=...)` for one caller) | No |
| Record what a feature is and where it lives | Call the `map_feature` tool (or run `/map survey`); the card lands in `.governor/map/` and in the system prompt of every new conversation and subagent (a running conversation keeps the map it started with; `/clear` picks the change up). See `src/governor/map.py` | No |
| Get JSON from a model | Set `output_schema` on the `AgentSpec` (or `output_schema:` in a profile); read `RunResult.structured`. The loop validates it with `agent_core.validate` | No |
| Show streamed output somewhere new | Consume `delta` events in a `Sink` (see `LogSink` in `tui.py`) | No |
| Consume run events (logging, learning, UI) | Implement a `Sink` (`handle(event)`), see `src/governor/sinks.py`; a feature registers its sinks in `Feature.sinks`. They hang off `session.sinks` (a `FanOut`), so a sink that raises is reported once and skipped, never ending the run; never append to `runtime.events.sinks` directly | No |
| Add a user-facing option | A slash command, never a CLI flag: a `Command` in `src/governor/commands.py` (a harness control) or in the feature's module, registered through its `Feature`. It is a function of `(session, arg)` that speaks through `session.io` and never touches a widget. One global command per feature; sub-commands go in its `Mode.commands` | No |
| Add a new way to read typed input (a mode, like `/qa`) | A `Mode` in `src/governor/modes/` (see `modes/qa.py`): its sub-commands, line handler, placeholder, status badge, entry hint and input title. The TUI reads `session.mode`; never add an `if self.<feature>` branch to `tui.py` | No |
| Change a type, interface, event shape, or the loop | **Stop.** This is a core change, see below | Yes |

If a feature seems to require a core change, do not make it as part of the
feature. Say so explicitly, explain what core is missing, and propose the
core change as a separate step.

## Hard rules

1. **Core is stdlib-only and never imports `governor`.** Enforced by `tests/test_boundaries.py`.
2. **Use core's public API.** Write `from agent_core import X`, never
   `from agent_core.agent import X`. The only exception is `agent_core.testing` in tests.
3. **History is append-only.** Never edit, reorder or drop earlier messages in a run.
   Prompt caching and model reasoning blocks (`ReasoningBlock`, passed back verbatim)
   depend on it. Inject dynamic context through `ContextProvider.before_turn`.
4. **Tools never crash the loop.** Raise `ToolError("message")` for expected failures;
   unexpected exceptions are caught and returned to the model as error results. Arguments
   are validated against the tool's schema before it runs, so a tool may trust their types.
5. **Tests are offline.** Use `agent_core.testing.ScriptedProvider` (scripted model
   responses), botocore's `Stubber`, or a fake SDK client. Never call AWS, Anthropic or OpenAI from tests.
   `tests/test_core/` may import only `agent_core`.
6. **Config comes from the TUI's saved settings, env vars or `.governor/` files**, not
   hard-coded values. Never commit `.env` or credentials. Tests set `GOVERNOR_HOME` to a
   temp dir so they never touch `~/.config/governor`.

## Architecture and abstraction

The code is three rings. Inner: `src/agent_core/`, the loop and its five ports (`Provider`,
`Tool`, `ContextProvider`, `PermissionPolicy`, `Sink`); it is a hexagon already and features
have not needed to edit it. Outer: the surfaces, `tui.py` and `cli.py`, later a server started
from the TUI. Between them is the application layer: `session.py` (`Session`, the conversation and the one run sequence), `ports.py`
(`Interaction`, the one port facing the human), `commands.py`, `modes/` (`Mode`, `QAMode`) and
`features.py` (`Feature`, `FEATURES`). `build_session()` in `runtime.py` composes them, and
`tui.py` and `cli.py` are two surfaces over the same `Session`. Before it, `GovernorApp` held
the conversation and the run sequence existed three times, so every feature edited `tui.py`
in 8–12 places.

These rules hold:

1. **One run sequence.** Do not copy the run-and-report sequence (sign-in check, trace, usage
   total, chat save) into a new method. Call `Session.run()` (or `ask()`); if a feature cannot,
   the missing seam goes into `Session` before the feature.
2. **Never approximate a missing core hook in a surface.** If the loop cannot do what a
   feature needs (suspend, per-run stop, path roots, a supplied `run_id`), name the hook and
   propose it as an additive core-change PR, then build the feature. `Stoppable` was written in
   `tui.py` because core had no stop seam and was deleted when `Runtime.stop` arrived. Compose
   per-run variation with `replace(runtime, ...)` in one place, never scattered.
3. **No feature flags in the TUI's dispatch, status or escape code.** A feature is a module
   plus registrations (tools, context providers, sinks, commands, a mode, a settings
   namespace). If it seems to need an `if self.<feature>` branch in `tui.py`, the missing seam
   belongs in the application layer; add the seam first. `/qa` was ten such branches for 88
   lines of logic; it is now `modes/qa.py` and `tui.py` has none.
4. **One global command per feature; sub-commands, settings and help are mode-scoped.**
   Routine actions print a terse status; a tradeoff is explained once, at the point of choice.
   A second global command from one feature needs a reason in the PR.
5. **Application code never imports a UI.** Anything that owns conversation state or runs an
   agent imports neither `textual` nor `rich` nor `governor.tui`/`governor.cli`; only `tui.py`
   imports `textual`. Results are data (`Turn`, `Resumed`, status fragments); the surface
   decides how to show them. Enforced by `tests/test_boundaries.py`.
6. **Two ports face the human.** Run events go out through a `Sink`; questions and notices
   go through `ports.Interaction` (`say`, `text`, `secret`, `choose`, `confirm`,
   `run_interactive`, `show_mode`, `clear`; keep it that small). Application code never reads
   `/dev/tty`, `sys.stdin.isatty()` or opens a modal, and a headless caller (`io=None`) runs
   without asking rather than refusing.
7. **Compose process-global inputs; never discover them.** `project_dir`, settings, the
   prompter, sinks and the stop event are passed in by the composition root
   (`runtime.py`). No module reads the process cwd, `isatty` or env vars on its own path except
   `config.py`.
8. **Feature state lives with the feature.** Keep it in the feature's own objects and a store
   under `.governor/<feature>/`; do not add an attribute per feature to the app, the session,
   `Settings` or `Chat`. Records link by id (`chat.id`, `run_id`) instead of growing a field
   per feature. Review question for every PR: did this add a field to a shared object?
9. **Keep layers small.** A module in the application layer past about 300 lines, or `tui.py`
   growing with a feature, is the earliest sign that a layer has become the place everything
   lands. Split by feature, not by adding options.
10. **Build an abstraction when a second user is on the table, not before.** Justify each new
    interface by a feature that needs it, not by taste.

## Changing core (`src/agent_core/`)

Only when the task explicitly calls for it. Then:

- Keep it in its **own PR**, with no feature work mixed in. CI fails unless the PR has
  the `core-change` label, and code owners must review it.
- Bump `agent_core.__version__`; bump `events.SCHEMA_VERSION` if an event's fields change.
- Prefer additive changes: new optional fields, new protocols, hooks with defaults.
- Update `src/agent_core/README.md` if the contract table changes.

## Code conventions

- Type-hint all signatures; `from __future__ import annotations` at the top of modules.
- Dataclasses for data, `Protocol` for interfaces. Keep the loop small; add behavior
  through the seams above instead of options on `Agent`.
- Tool docstrings are the model's instructions: first paragraph = description,
  `Args:` section = parameter descriptions. Write them for the model.
- Match the surrounding style: short module docstrings explaining *why*, few inline comments.
- **UI text: explain once, where the choice is made.** Routine actions (a task, a `/qa`
  question, a stop) print at most a short status line. A tradeoff or how-to belongs where
  the user makes the decision: picker option labels, the command's `/help` line, a single
  line right after the setting changes, or `USER_GUIDE.md`. It never repeats on every use
  or every launch. Errors are the exception: keep the actionable hint (`ProviderError(hint=...)`),
  and don't add a generic one on top.
- **Features stay in their place.** The harness's global surfaces belong to its core
  controls (provider, model, effort, agent, sign-in, conversation, cost): the command list,
  `/help`, the startup banner, the status bar, and anything printed on every prompt or turn.
  A feature you add (Q&A, the project map, …) gets **one** entry in the command
  list, plus one row in the README/USER_GUIDE command tables, and nothing else up front:
  - A feature with several commands runs as a mode (see `/qa`). Its other commands live in
    its `Mode.commands`, exist only while it's on, and may reuse generic verbs (`/model`,
    `/done`) that mean "for this mode" there, as Q&A does. Never add global
    `/<feature>-model`, `/<feature>-off`, … commands.
  - Its settings are chosen inside it, and its tradeoffs are explained there, once.
  - Its keys and tips show in `/help` only while it's on. No footer lines and no hints
    on launch. A status-bar marker is fine only while it's on (`Q&A`).
  - Nothing it does is printed on every prompt or turn unless the user asks for it
    (an inspect command such as `/context`).
  - Its user docs get their own subsection, not paragraphs inside the general sections.

## Map of key files

```
src/agent_core/agent.py        the loop: Agent.run(), Runtime, AgentSpec
src/agent_core/tools.py        @tool, schema generation, ToolRegistry.execute()
src/agent_core/subagent.py     `task` tool (subagents, parallel when called together)
src/agent_core/context.py      ContextProvider protocol
src/agent_core/events.py       event schema (trace format)
src/agent_core/schema.py       JSON Schema validation for tool arguments and structured output
src/agent_core/provider.py     Provider protocol: complete(request, on_delta=, stop=), ProviderCancelled
src/governor/runtime.py         build_runtime() and build_session(): the composition roots
src/governor/session.py         Session: the conversation and the one run sequence; Turn, Resumed
src/governor/ports.py           Interaction: the one port facing the human (implemented by tui.py)
src/governor/commands.py        Command and the harness's slash commands, functions of (session, arg)
src/governor/modes/             Mode protocol; qa.py is /qa (QASession must keep its prompt prefix identical)
src/governor/features.py        Feature descriptor and FEATURES (builtin, qa, chats, map, budget)
src/governor/budget.py          /budget: a cap on the session's spend; a sink sums model_call cost and stops the run at the cap
src/governor/providers/bedrock.py   Bedrock Converse provider + error hints
src/governor/providers/mantle.py    Claude Messages API on Bedrock; translation shared with anthropic_api.py
src/governor/providers/openai_api.py  OpenAI Responses API (stateless, encrypted reasoning)
src/governor/models.py          per-provider model tables, roles (main/subagent), prices, resolve()
src/governor/effort.py          /effort: one level across providers, merged into each call by wrapping the provider
src/governor/config.py          ~/.config/governor settings + saved credentials
src/governor/auth.py            /login flows and sign-in status per provider
src/governor/tui.py             full-screen terminal UI (Textual) over a Session; tests drive it headlessly with run_test()
src/governor/chats.py           saved chats (.governor/chats/) and the /chats command
src/governor/map.py             the project map (.governor/map/): area and feature cards, the map_feature tool, MapSink (confirmed paths), MapContext, /map
src/governor/cli.py             `govern`: the TUI, and nothing else (no flags, no subcommands)
tests/test_core/               core tests (agent_core only)
tests/test_governor/            Governor tests (test_session.py drives the Session headlessly)
tests/test_boundaries.py       dependency rules
```
