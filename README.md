# Governor

> **New here?** Read [USER_GUIDE.md](USER_GUIDE.md) for setup and everyday use. Coding agents: see [AGENTS.md](AGENTS.md).

Governor is a small, modular agent harness for Amazon Bedrock, the Anthropic API and the OpenAI API,
with a full-screen terminal UI. It has an agent loop with
tools, parallel subagents, markdown agent profiles, streamed replies that can be
cancelled mid-call, schema-checked tool calls and JSON output, saved chats that can be
resumed, JSONL traces with per-run cost, a project map the agent surveys and keeps
(`/map`), a spending cap (`/budget`), and one seam where a knowledge or retrieval layer can inject context. It is about 6.7k lines of Python on top of the providers'
SDKs, `pyyaml` and `textual`.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
govern                          # opens the terminal UI
```

`govern` is the only command you need. The first run asks for a provider and signs
you in; after that it opens straight to a prompt. Everything else is a slash command:

| Command | Does |
| --- | --- |
| `/provider` | Pick where models run: `bedrock` (Converse, any family), `mantle` (Claude via the Messages API on Bedrock), `anthropic` (Claude API), `openai` (OpenAI Responses API) |
| `/model` | One picker for the main, subagent and Q&A models from **that provider's** table (←/→ switch between them); remembered per provider. `/model sol` sets the main model directly |
| `/effort` | How hard the model thinks: `low`, `medium`, `high`, `max`, or back to the model's default. One scale for every provider (Claude `output_config.effort`, OpenAI `reasoning.effort`); Shift-Tab cycles it, Ctrl-0…4 jump |
| `/login`, `/logout` | API key for Anthropic/OpenAI; `aws login`, `aws sso login` or a Bedrock API key for AWS |
| `/agent` | Pick the agent profile |
| `/doctor` | Check credentials and ping the current models, with fix-it hints |
| `/qa` | Ask questions about the conversation without adding to it: each starts fresh, `+` follows up, Esc or `/done` leaves. Inside Q&A, `/model` picks the answering model; the conversation's own by default, which reuses its prompt cache |
| `/done` (Q&A only) | Leave Q&A and return to the conversation |
| `/chats` | List the project's saved chats and resume one. Every completed turn is saved to `.governor/chats/` |
| `/map` | The project map the agent reads at the start of every run (about 3k tokens of area and feature cards, stored in `.governor/map/`): `/map` shows it, `/map survey` builds it with an explore run, `/map refresh` updates it, `/map <feature>` shows one card with its predicted and confirmed paths |
| `/budget` | Cap what the session spends: a run stops after the model call that reaches it, with the conversation as it was. `/budget 2` sets $2, `/budget off` clears it; saved across sessions |
| `/clear`, `/cost`, `/yolo`, `/verbose`, `/help` | Conversation and display controls |
| `/exit` | Quit Governor (or press Ctrl-D on an empty input) |

Settings live in `~/.config/governor/settings.json` and saved keys in
`~/.config/governor/credentials` (mode 600, loaded at startup). Shell environment
variables and a project `.env` take precedence over saved keys, so the standard
`AWS_PROFILE`, `AWS_BEARER_TOKEN_BEDROCK`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` all still work.

An agent's `params` are passed to the provider untouched, so they are provider-specific:
`output_config: {effort: high}` for Claude, `reasoning: {effort: high}` for OpenAI. For
the `anthropic` provider, `params.betas` sends the call through the beta Messages
endpoint (e.g. `betas: [server-side-fallback-2026-07-01]` with `fallbacks: default`).

If a model fails, `/doctor` prints a hint. The usual Bedrock causes are:

| Symptom | Fix |
| --- | --- |
| "on-demand throughput isn't supported" | Use the `us.`/`global.` inference-profile id; `/model` → *Other model id…* lists the ones your account has |
| AccessDenied mentioning model access or a use case | Bedrock console → Model access → enable the model and submit the Anthropic use-case form (once per account) |
| AccessDenied without that mention | IAM needs `bedrock:InvokeModel*` on `inference-profile/*` **and** `foundation-model/*` in every region the profile routes to |
| Id not in this region | Pick another with `/model`, or re-point the entry in `.governor/models.toml` |

### Library use

`govern` is the only command: there are no flags, subcommands or scripted mode. From code,
build a session; it never prompts (tools that need approval are refused unless allowed with
`GOVERNOR_ALLOW=bash,write_file`, or `*`), and every session writes a trace to `.governor/runs/`.

```python
from pathlib import Path
from governor import build_session

session = build_session(Path("."))  # headless: never asks; io=... adds a human port
turn = session.ask("What does this repo do?")
print(turn.text, session.total.cost_usd)
session.close()
```

`build_runtime(Path("."))` plus `agent_core.Agent` still gives the bare loop without a session.

## Layout: primitives vs governor

The code is two packages with a one-way dependency: `governor` → `agent_core`.

```
src/agent_core/      THE PRIMITIVES: stdlib only, changes only via `core-change` PRs
  types.py           provider-neutral messages/blocks/usage (shaped like Converse)
  provider.py        Provider protocol, ProviderError
  models.py          ModelInfo (provider, id, prices, cost())
  tools.py           Tool, @tool, schema generation, ToolRegistry, ToolContext
  permissions.py     PermissionPolicy protocol, StaticPolicy
  context.py         ContextProvider protocol: where memory and retrieval plug in
  events.py          Event schema, EventBus, Sink
  schema.py          JSON Schema validation (tool arguments, structured output), stdlib only
  agent.py           AgentSpec, Runtime, Agent.run(): the loop and its three policies
  subagent.py        the `task` delegation tool
  testing.py         ScriptedProvider for offline tests

src/governor/         GOVERNOR: concrete choices, where feature work happens
  runtime.py         build_runtime() and build_session(): the composition roots
  session.py         Session: the conversation and the one run sequence (Turn, Resumed)
  ports.py           Interaction: the one port facing the human; tui.py implements it
  commands.py        Command and the harness's slash commands, functions of (session, arg)
  modes/             Mode protocol · qa.py: /qa, questions branched off the conversation
  features.py        Feature descriptor and FEATURES (builtin, qa, chats, map, budget): one entry per feature
  providers/         bedrock.py (Converse, any model family) · mantle.py (Claude via Messages API on Bedrock)
                     anthropic_api.py (Claude API) · openai_api.py (OpenAI Responses API)
  models.py          per-provider model tables (main/subagent roles) + $/Mtok; .governor/models.toml overrides
  effort.py          /effort: one level for every provider, added to each call by wrapping the provider
  config.py, auth.py saved settings and credentials; /login flows per provider
  tui.py             the terminal UI (Textual) over a Session: prompt, pickers, the live reply
  chats.py           saved chats in .governor/chats/ and the /chats command
  map.py             the project map in .governor/map/: area and feature cards, the map_feature tool,
                     MapSink (the paths a run actually touched), MapContext (the map in the system prompt), /map
  budget.py          /budget: a cap on the session's spend, enforced by a sink that stops the run
  tools/             files.py · shell.py · external.py (any-language tools)
  agents/*.md        built-in profiles: general, explore, planner
  context.py         EnvironmentContext, AgentsMdContext
  permissions.py     interactive allow / ask / deny
  sinks.py           FanOut (the one sink on the bus; a raising subscriber is skipped), JSONL trace, console printer
  doctor.py, cli.py (`govern` → the TUI, nothing else), profiles.py
```

Why: feature PRs (a new tool, a prompt tweak, a provider, a knowledge-base context
provider) touch only `governor/`, so reviewers can see at a glance that the contracts
everything depends on are unchanged. How it is enforced:

- `tests/test_boundaries.py`: core imports only the stdlib; governor imports core only
  through `from agent_core import ...`; only `tui.py` imports `textual`/`rich`, and the
  application layer never imports a surface.
- CI (`core-guard`): a PR touching `src/agent_core/` fails unless labelled `core-change`.
- `.github/CODEOWNERS`: core changes need a core owner's review (fill in the handles).
- `src/agent_core/README.md`: what belongs in core, how to change it, and the 0.x → 1.0
  stability plan. While on 0.x, moving things between the packages is expected.

## Interfaces

Every public class (its fields and public methods) and module-level function under `src/`, generated
from the source by `python tests/test_interfaces.py`; `pytest` fails when this section drifts.
Protocols are the ports a feature implements; dataclasses are the data that crosses them.

<details>
<summary>All interfaces, by module</summary>

<!-- interfaces:start -->

**`src/agent_core/agent.py`**

- `AgentSpec` (dataclass): fields name, description, system_prompt, model, tools, max_turns, max_tokens, subagent, permissions, params, output_schema, max_invalid_calls
- `Runtime` (dataclass): fields registry, profiles, resolve_model, provider_for, events, permissions, context_providers, max_depth, max_parallel_tools, model_override, stop
- `RunResult` (dataclass): fields text, stop, messages, usage, run_id, turns, structured
- `Agent`: `__init__(spec, runtime, *, depth=0, parent_run_id=None, cwd=None)` · `run(task, history=None) -> RunResult`

**`src/agent_core/context.py`**

- `RunView` (dataclass): fields spec, task, cwd, depth, turn, messages
- `ContextProvider` (protocol): `system(run) -> str | None` · `before_turn(run) -> str | None`

**`src/agent_core/events.py`**

- `Event` (dataclass): fields type, run_id, agent, depth, parent_run_id, data, ts, v · `to_json() -> str`
- `Sink` (protocol): `handle(event) -> None`
- `EventBus`: `__init__(sinks=None)` · `emit(event) -> None`
- `Collector`: `__init__()` · `handle(event) -> None` · `of_type(type_) -> list[Event]`

**`src/agent_core/models.py`**

- `ModelInfo` (dataclass): fields alias, provider, model_id, input_per_mtok, output_per_mtok, cache_read_per_mtok, cache_write_per_mtok, supports_cache · `cost(usage) -> float`

**`src/agent_core/permissions.py`**

- `PermissionPolicy` (protocol): `check(tool, args, default, profile=None) -> str | None`
- `StaticPolicy`: `__init__(overrides=None, *, yolo=False)` · `check(tool, args, default, profile=None) -> str | None`

**`src/agent_core/provider.py`**

- `ProviderError` · RuntimeError: `__init__(message, *, hint='', retryable=False)`
- `ProviderCancelled` · ProviderError: `__init__(partial=None)`
- `Provider` (protocol): fields name · `complete(request, *, on_delta=None, stop=None) -> ModelResponse`

**`src/agent_core/schema.py`**

- `validate(instance, schema, *, root=None, path='') -> list[str]`
- `parse_json(text) -> tuple[Any, str | None]`

**`src/agent_core/testing.py`**

- `text(s, usage=None) -> ModelResponse`
- `tool_call(name, args=None, *, say='', usage=None) -> ModelResponse`
- `tool_calls(calls, *, say='', usage=None) -> ModelResponse`
- `ScriptedProvider`: `__init__(script=(), router=None)` · `complete(request, *, on_delta=None, stop=None) -> ModelResponse`

**`src/agent_core/tools.py`**

- `ToolContext` (dataclass): fields runtime, cwd, agent, run_id, depth, add_usage · `resolve(path) -> Path`
- `Tool` (dataclass): fields name, description, input_schema, fn, permission, spec_fn · `spec(runtime) -> ToolSpec`
- `schema_from_function(fn) -> tuple[str, dict[str, Any]]`
- `tool(fn=None, *, name=None, permission='ask') -> Any`
- `ToolRegistry` (dataclass): fields tools · `register(*tools) -> ToolRegistry` · `names(patterns) -> list[str]` · `execute(call, ctx, allowed, profile_permissions) -> ToolResultBlock` · `run(call, ctx, allowed, profile_permissions) -> tuple[ToolResultBlock, Outcome]`
- `ToolError` · Exception

**`src/agent_core/types.py`**

- `StopReason` · StrEnum
- `TextBlock` (dataclass): fields text, type
- `ToolUseBlock` (dataclass): fields id, name, input, type
- `ToolResultBlock` (dataclass): fields tool_use_id, content, is_error, type
- `ReasoningBlock` (dataclass): fields raw, provider, type
- `Message` (dataclass): fields role, content · `user(text) -> Message` · `text() -> str` · `tool_uses() -> list[ToolUseBlock]` · `to_dict() -> dict[str, Any]` · `from_dict(data) -> Message`
- `block_to_dict(block) -> dict[str, Any]`
- `block_from_dict(data) -> Block`
- `ToolSpec` (dataclass): fields name, description, input_schema
- `Usage` (dataclass): fields input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd · `to_dict() -> dict[str, Any]`
- `ModelRequest` (dataclass): fields model, messages, system, tools, max_tokens, temperature, cache, params, output_schema
- `Delta` (dataclass): fields kind, text, index
- `ModelResponse` (dataclass): fields message, stop_reason, usage, model, latency_s, raw

**`src/governor/auth.py`**

- `Prompter` (protocol): `say(text) -> None` · `text(message, default='') -> str` · `secret(message) -> str` · `choose(message, options) -> str | None` · `run_interactive(command) -> int`
- `Status` (dataclass): fields ok, detail
- `status(provider) -> Status`
- `login(provider, ui) -> bool`
- `logout(provider) -> list[str]`

**`src/governor/budget.py`**

- `parse_cap(value) -> float | None`
- `dollars(amount) -> str`
- `BudgetSink`: `__init__(session)` · `cap() -> float | None` · `handle(event) -> None`
- `sinks(session) -> list[BudgetSink]`
- `status(session) -> list[tuple[str, str]]`
- `args(session) -> list[str]`
- `command(session, arg) -> None`

**`src/governor/chats.py`**

- `title_from(task, limit=60) -> str`
- `Chat` (dataclass): fields id, title, provider, model, agent, messages, usage, created, updated · `new(task, provider, model, agent) -> Chat` · `to_dict() -> dict[str, Any]` · `from_dict(data) -> Chat` · `turns() -> int`
- `ChatStore`: `__init__(project_dir)` · `path(chat_id) -> Path` · `save(chat) -> Path` · `load(chat_id) -> Chat` · `list(limit=30) -> list[Chat]`
- `resume_command(session, arg) -> None`

**`src/governor/cli.py`**

- `main(argv=None) -> int`

**`src/governor/commands.py`**

- `Command` (dataclass): fields name, help, run, args, echo
- `model_names(session) -> list[str]`
- `ask_model_id(session, provider) -> str`
- `pick_models(session, start='main') -> tuple[str, str] | None`
- `set_role_model(session, role, name) -> ModelInfo | None`
- `set_effort(session, level) -> str`
- `provider(session, arg) -> None`
- `model(session, arg) -> None`
- `effort(session, arg) -> None`
- `login(session, arg) -> None`
- `logout(session, arg) -> None`
- `agent(session, arg) -> None`
- `doctor_command(session, _) -> None`
- `clear(session, _) -> None`
- `cost(session, _) -> None`
- `yolo(session, _) -> None`

**`src/governor/config.py`**

- `home() -> Path`
- `read_env_file(path) -> dict[str, str]`
- `load_environment(project_dir) -> None`
- `saved_credentials() -> dict[str, str]`
- `save_credentials(values) -> Path`
- `Settings` (dataclass): fields provider, models, agent, effort, features · `load() -> Settings` · `save() -> None` · `of(feature) -> dict[str, Any]` · `selection() -> Selection` · `effort_level() -> str | None` · `set_model(role, name) -> None`

**`src/governor/context.py`**

- `EnvironmentContext`: `system(run) -> str | None` · `before_turn(run) -> str | None`
- `AgentsMdContext`: `__init__(filename='AGENTS.md')` · `system(run) -> str | None` · `before_turn(run) -> str | None`
- `default_context_providers() -> list[ContextProvider]`

**`src/governor/doctor.py`**

- `list_inference_profiles() -> list[dict]`
- `agreement_status(model_id) -> str | None`
- `ping(info) -> tuple[bool, str]`
- `doctor(provider, models, out=print) -> int`

**`src/governor/effort.py`**

- `params_for(level, info) -> dict[str, Any]`
- `merged(params, extra) -> dict[str, Any]`
- `EffortProvider`: `__init__(inner, info, level)` · `complete(request, *, on_delta=None, stop=None) -> ModelResponse`
- `wrap(provider_for, level) -> Callable[[ModelInfo], Provider]`

**`src/governor/features.py`**

- `Feature` (dataclass): fields name, tools, context_providers, sinks, commands, status, close

**`src/governor/map.py`**

- `Area` (dataclass): fields path, purpose, contains, depends_on · `from_dict(data) -> Area`
- `FeatureCard` (dataclass): fields name, abstract, paths, read, wrote, runs, chats, updated · `from_dict(data) -> FeatureCard`
- `Survey` (dataclass): fields at, commit, model
- `ProjectMap` (dataclass): fields areas, features, surveyed · `empty() -> bool` · `feature(name) -> FeatureCard | None` · `to_dict() -> dict[str, Any]` · `from_dict(data) -> ProjectMap`
- `MapStore`: `__init__(project_dir)` · `errors() -> list[str]` · `of(project_dir) -> MapStore` · `load() -> ProjectMap` · `save(project_map) -> None` · `update(change, *, rewrite=False) -> T` · `relative(path) -> str | None` · `map_feature(name, paths, abstract) -> FeatureCard` · `confirm(name, read, wrote, run_id, chat_id) -> None` · `fill(data, *, commit, model) -> ProjectMap`
- `estimate(text) -> int`
- `tree(areas) -> list[tuple[int, Area]]`
- `render(project_map, budget=BUDGET_TOKENS, *, note=False) -> str`
- `can_map(spec) -> bool`
- `opens_with(history, prefix) -> bool`
- `MapContext`: `__init__()` · `system(run) -> str | None` · `before_turn(run) -> str | None`
- `feature_tool(store) -> Tool`
- `MapSink`: `__init__(store, chat_id=lambda: None)` · `handle(event) -> None`
- `survey_spec(profiles) -> AgentSpec`
- `refresh_prompt(current, changes) -> str`
- `git_head(project_dir) -> str | None`
- `changes_since(project_dir, commit) -> str | None`
- `status_line(m) -> str`
- `map_command(session, arg) -> None`
- `tools(project_dir) -> list[Tool]`
- `context_providers() -> list[ContextProvider]`
- `sinks(session) -> list[Sink]`

**`src/governor/models.py`**

- `short_name(info) -> str`
- `catalog(project_dir=None) -> dict[str, dict[str, ModelInfo]]`
- `Selection` (dataclass): fields provider, roles · `default(provider) -> Selection`
- `resolve(name, selection, project_dir=None) -> ModelInfo`

**`src/governor/modes/__init__.py`**

- `Mode` (protocol): fields name, placeholder, hint, commands · `submit(session, line) -> None` · `leave(session) -> None` · `title(draft) -> str | None`

**`src/governor/modes/qa.py`**

- `NoTools`: `check(tool, args, default, profile=None) -> str | None`
- `QASession` (dataclass): fields base, spec, runtime, model, thread · `start(history, spec, runtime, model, conversation_model) -> QASession` · `request(question, follow_up=False) -> tuple[list[Message], str]` · `record(result) -> None`
- `QAMode`: `__init__(session)` · `hint() -> str` · `pick_model(session, arg) -> None` · `submit(session, line) -> None` · `leave(session) -> None` · `title(draft) -> str | None`
- `enter(session, arg) -> None`

**`src/governor/permissions.py`**

- `asker_from(io, stop=None) -> Asker | None`
- `Permissions`: `__init__(overrides=None, *, yolo=False, asker=None)` · `decide(tool, default, profile=None) -> Decision` · `check(tool, args, default, profile=None) -> str | None`

**`src/governor/ports.py`**

- `Tab` (dataclass): fields id, label, options, default
- `Interaction` (protocol): `say(text, kind='plain') -> None` · `text(message, default='', *, suggestions=None) -> str` · `secret(message) -> str` · `choose(message, options, default=None) -> str | None` · `choose_tabbed(message, tabs, start) -> tuple[str, str] | None` · `confirm(message, *, yes='Yes', no='No') -> bool` · `run_interactive(command) -> int` · `show_mode(mode) -> None` · `clear() -> None`

**`src/governor/profiles.py`**

- `parse_profile(text, source='<string>') -> AgentSpec`
- `load_profiles(project_dir=None) -> dict[str, AgentSpec]`

**`src/governor/providers/__init__.py`**

- `get_provider(name) -> Provider`
- `reset_providers() -> None`

**`src/governor/providers/anthropic_api.py`**

- `AnthropicProvider`: `__init__(client=None)` · `complete(request, *, on_delta=None, stop=None) -> ModelResponse`

**`src/governor/providers/bedrock.py`**

- `aws_session() -> boto3.Session`
- `explain_error(exc) -> ProviderError`
- `BedrockProvider`: `__init__(client=None)` · `complete(request, *, on_delta=None, stop=None) -> ModelResponse`
- `to_converse(request) -> dict[str, Any]`
- `from_converse(resp, model, latency_s) -> ModelResponse`
- `collect_stream(events, model, started, on_delta, stop) -> ModelResponse`

**`src/governor/providers/mantle.py`**

- `MantleProvider`: `__init__(client=None)` · `complete(request, *, on_delta=None, stop=None) -> ModelResponse`
- `to_messages_api(request, provider=NAME) -> dict[str, Any]`
- `from_messages_api(resp, model, latency_s, provider=NAME) -> ModelResponse`
- `run_stream(api, kwargs, model, provider, on_delta, stop) -> ModelResponse`

**`src/governor/providers/openai_api.py`**

- `OpenAIProvider`: `__init__(client=None)` · `complete(request, *, on_delta=None, stop=None) -> ModelResponse`
- `to_responses_api(request) -> dict[str, Any]`
- `from_responses_api(resp, model, latency_s) -> ModelResponse`

**`src/governor/runtime.py`**

- `build_runtime(project_dir, selection=None, effort=None, **overrides) -> Runtime`
- `build_session(project_dir, *, io=None, sinks=(), settings=None, features=FEATURES, save_chats=True, trace_label='session', **overrides) -> Session`

**`src/governor/session.py`**

- `summary(usage, seconds=None) -> str`
- `strip_reasoning(messages) -> list[Message]`
- `Turn` (dataclass): fields result, model, seconds, error, stopped · `ran() -> bool` · `ok() -> bool` · `text() -> str` · `usage() -> Usage` · `status() -> str`
- `Resumed` (dataclass): fields chat, messages, switched, note
- `Session`: `__init__(project_dir, settings, runtime, io, commands, chats=None, *, sinks=None, status=(), trace_label='session', closers=())` · `say(text, kind='plain') -> None` · `choose(message, options, default=None) -> str | None` · `input(message, default='', *, suggestions=None) -> str` · `offer_sign_in(provider) -> bool` · `sign_in(provider) -> bool` · `spec() -> AgentSpec` · `run(spec, prompt, *, history=None, runtime=None, quiet=False) -> Turn` · `ask(task) -> Turn` · `request_stop(by='user') -> None` · `close() -> None` · `resume(chat) -> Resumed` · `clear() -> None` · `enter_mode(mode) -> None` · `leave_mode() -> None` · `submit(line) -> None` · `status() -> list[tuple[str, str]]`

**`src/governor/sinks.py`**

- `FanOut`: `__init__(sinks=(), on_error=None)` · `add(sink) -> None` · `remove(sink) -> None` · `find(kind) -> S | None` · `handle(event) -> None`
- `JsonlTrace`: `__init__(path)` · `handle(event) -> None` · `close() -> None`
- `summarize_calls(calls, limit=6) -> str`
- `ConsolePrinter`: `__init__(stream=sys.stderr, verbose=False, write=None)` · `handle(e) -> None`

**`src/governor/tools/__init__.py`**

- `builtin_tools(project_dir=None) -> list[Tool]`
- `default_registry(project_dir=None) -> ToolRegistry`

**`src/governor/tools/external.py`**

- `load_external_tools(directory) -> list[Tool]`

**`src/governor/tools/files.py`**

- `read_file(ctx, path, offset=1, limit=2000) -> str`
- `write_file(ctx, path, content) -> str`
- `edit_file(ctx, path, old_string, new_string, replace_all=False) -> str`
- `list_files(ctx, pattern='**/*', path='.') -> str`
- `grep(ctx, pattern, path='.', glob=None, ignore_case=False) -> str`

**`src/governor/tools/shell.py`**

- `bash(ctx, command, timeout=120) -> str`

**`src/governor/tui.py`**

- `LiveReply`: `__init__(app, classes)` · `add(piece) -> None` · `finish(text) -> None`
- `LogSink`: `__init__(app)` · `handle(event) -> None`
- `ChoiceScreen` · ModalScreen[Any]: fields BINDINGS · `__init__(title, options=None, default=None, *, tabs=(), start=None)` · `compose() -> ComposeResult` · `on_mount() -> None` · `action_tab(step) -> None` · `on_option_list_option_selected(event) -> None` · `action_cancel() -> None`
- `TextScreen` · ModalScreen[str]: fields BINDINGS · `__init__(title, default='', secret=False, suggestions=None)` · `compose() -> ComposeResult` · `on_mount() -> None` · `on_input_submitted(event) -> None` · `action_cancel() -> None`
- `PromptArea` · TextArea: `__init__(intercept, **kwargs)`
- `GovernorApp` · App[None]: fields BINDINGS · `__init__(project_dir)` · `compose() -> ComposeResult` · `on_mount() -> None` · `target() -> Widget` · `say(text, kind='plain') -> None` · `clear() -> None` · `show_mode(mode) -> None` · `text(message, default='', *, suggestions=None) -> str` · `secret(message) -> str` · `choose(message, options, default=None) -> str | None` · `choose_tabbed(message, tabs, start) -> tuple[str, str] | None` · `confirm(message, *, yes='Yes', no='No') -> bool` · `run_interactive(command) -> int` · `on_text_area_changed(event) -> None` · `on_prompt_area_submitted(event) -> None` · `action_interrupt() -> None` · `action_escape() -> None` · `action_quit_if_empty() -> None` · `action_effort(level) -> None` · `action_cycle_effort() -> None`

<!-- interfaces:end -->

</details>

## Extending it

**Add a tool** by writing a typed function. The schema is generated from its signature and docstring.

```python
from agent_core import ToolContext, tool


@tool(permission="allow")
def word_count(ctx: ToolContext, path: str) -> str:
    """Count words in a file.

    Args:
        path: file to count.
    """
    return str(len(ctx.resolve(path).read_text().split()))


runtime.registry.register(word_count)
```

**Add a tool in any language** (TypeScript, Go and so on) by putting a manifest in
`.governor/tools/<name>.json` with `name`, `description`, `input_schema`, `command`
and `permission`. The command receives the tool input as JSON on stdin and prints its
result. See `src/governor/tools/external.py`.

**Add or override an agent** by putting a markdown file in `.governor/agents/<name>.md`.
The frontmatter fields are `name`, `description`, `model` (`main`, `subagent`, a model name
in the active provider, or `provider/model`), `tools` (with `"*"` and
`-name` exclusions), `max_turns`, `subagent`, `permissions` and `params`. The body is
the system prompt.

**Change models**: pick them in the TUI, or add/override table entries in
`.governor/models.toml`, one table per provider:

```toml
[bedrock.qwen]
model_id = "qwen.qwen3-coder-480b-a35b-v1:0"
input_per_mtok = 0.22
output_per_mtok = 1.80
supports_cache = false
```

**Inject knowledge** by implementing `ContextProvider` with two methods:

- `system(run)` runs once, and its text goes into the system prompt, which is part of the cached prefix.
- `before_turn(run)` runs before every model call, and its text is added to the newest
  user message as `<context>`. Use it for retrieved memories or filtered previews.

History is never edited, so prompt caching keeps working, and the trace records what
was injected and when.

**Get JSON back** by giving a profile an `output_schema` (JSON Schema). The request uses
the provider's native structured-output mode (Converse `outputConfig`, Messages
`output_config.format`, Responses `text.format`), and the loop still parses and validates
the reply: a reply that doesn't match gets one corrective turn, then the run ends with
`invalid_output`. The parsed value is `RunResult.structured`.

```python
spec = replace(
    runtime.profiles["explore"],
    output_schema={"type": "object", "properties": {"files": {"type": "array"}}, "required": ["files"]},
)
print(Agent(spec, runtime).run("list the entry points").structured)
```

**Add a provider** by implementing `complete(request, *, on_delta=None, stop=None) -> ModelResponse`
(the `agent_core.Provider` protocol) and registering it in `src/governor/providers/__init__.py`.
Call `on_delta` with each streamed fragment when you can; raise `ProviderCancelled` when `stop`
is set mid-call. A provider that ignores both still works, just not live.

## Design notes

- **State and calls.** Run state is an append-only message list, and each model call moves
  it forward. Never editing history keeps prompt caching effective and keeps reasoning
  blocks valid. Recent Claude models require their thinking blocks to be sent back unchanged,
  and `ReasoningBlock` carries them through untouched.
- **Streaming and stopping.** Every provider streams, and the loop re-emits fragments as `delta`
  events (text, reasoning, tool input) for the TUI and any other sink; the JSONL
  trace drops them because the full message follows in `model_call`. `Runtime.stop` is one event
  for the whole tree: when set, the model call in flight is cancelled (the connection is closed),
  pending tool calls are refused, and each run ends with `cancelled`. The TUI then leaves the
  conversation as it was before that message.
- **Tool calls are checked before they run.** Arguments are validated against the tool's JSON
  schema (types, required and unexpected keys, enums, bounds) and the model gets a precise error
  instead of a Python traceback. A tool called wrongly `max_invalid_calls` times in a row ends the
  run with `invalid_tool_calls` rather than looping. A reply cut off at `max_tokens` with tool
  calls in it has them refused with a note to retry smaller.
- **Subagents.** A subagent is a tool call that runs a fresh `Agent` at `depth+1`, limited by
  `max_depth` (default 2). Several `task` calls in one response run in parallel. A child's usage rolls
  up into its parent's, and its events share the parent's trace with a `parent_run_id`.
- **Synchronous with threads.** boto3 is synchronous, and threads give parallel tools without
  async plumbing.
- **Traces.** `run_start` records the system prompt, tools and model. Each `model_call` records
  the full assistant message, and each `tool_result` records the full output. Together these
  are enough to rebuild a run for replay or evaluation.
- **Model routing.** Profiles name a role, not a vendor model: planner and general use
  `main`, explore uses `subagent`. Switching provider moves every agent at once; setting the
  subagent model to the main one (the Subagent page of `/model`) runs the whole tree on one model for A/B runs.

## Not built yet (and where each would go)

| Feature | Where it would go |
| --- | --- |
| Context compaction | A `ContextProvider` combined with a summarizing subagent |
| Chat manager (connect prompts to past chats and project context) | A `ContextProvider` over saved chats and the project map |
| Knowledge base | A `ContextProvider` whose `before_turn` injects retrieved context |
| MCP client | Becomes a source of `Tool`s |
| HTTP API | A server over `build_session(io=<question queue>, sinks=[<event stream>])`, started from inside the TUI (no new entry point) |
| Richer UI (e.g. TypeScript) | Consume that event stream; the commands in `commands.py` already run without the TUI |

## Tests

```bash
pytest          # offline: ScriptedProvider + botocore Stubber, no AWS calls
                # tests/test_core (core only) · tests/test_governor · tests/test_boundaries.py
ruff check .
```
