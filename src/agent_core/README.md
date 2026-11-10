# agent_core: the agent primitives

`agent_core` holds the contracts every part of the system builds on, plus the
reference agent loop. It is deliberately small, standard-library only, and slow to
change. `governor` (and anything else, such as a knowledge-base package or an
experimental runner) builds on top of it.

## What lives here

| Module | Contract |
| --- | --- |
| `types.py` | Messages, content blocks, `ModelRequest`/`ModelResponse` (with `output_schema`), `Delta`, `Usage`, `ToolSpec` |
| `provider.py` | `Provider` protocol (`complete(request, *, on_delta, stop)`), `ProviderError`, `ProviderCancelled` |
| `models.py` | `ModelInfo` (provider, id, prices, `cost()`) |
| `tools.py` | `Tool`, `@tool`, schema generation, `ToolRegistry`, `ToolContext`, `ToolError` |
| `permissions.py` | `PermissionPolicy` protocol, `Decision`, `StaticPolicy` |
| `context.py` | `ContextProvider` protocol (`system`, `before_turn`), `RunView` |
| `schema.py` | `validate(instance, schema)` and `parse_json(text)`: the JSON Schema subset the loop enforces |
| `events.py` | `Event` schema (+ `SCHEMA_VERSION`), `EventBus`, `Sink`, `Collector` |
| `agent.py` | `AgentSpec` (`output_schema`, `max_invalid_calls`), `Runtime` (`stop`), `Agent.run()` (the loop), `RunResult` (`structured`) |
| `subagent.py` | the `task` delegation tool |
| `testing.py` | `ScriptedProvider` and response helpers for offline tests |

## What does not live here

Anything with a concrete choice in it goes in `governor`: Bedrock or other providers,
model ids and prices, file/shell tools, agent profiles, prompts, CLI, terminal
output, interactive permissions, AGENTS.md loading, retrieval/knowledge-base code.

Rule of thumb: if two reasonable people could want different versions of it, it is
governor. If every component has to agree on it, it is core.

## Rules (enforced by `tests/test_boundaries.py`)

1. **Stdlib only.** Core imports nothing outside the standard library and itself.
2. **Core never imports `governor`.** Dependencies point one way.
3. **Public API only.** Code outside core imports from `agent_core` (this package's
   `__init__`), never `agent_core.<module>` (except `agent_core.testing`). That lets
   core reorganize its files without breaking anyone.

## Changing core

- A PR that touches `src/agent_core/` is a **core-change PR**. It needs the
  `core-change` label (CI fails without it) and a review from a code owner.
- Keep core-change PRs separate from feature PRs. If a feature needs a new
  capability from core, land the core change first, on its own.
- Bump `agent_core.__version__`. Bump `events.SCHEMA_VERSION` if an event's shape
  changes, since stored traces depend on it.
- Prefer extending over changing: new optional fields, new protocols, or new hooks
  with defaults, so existing governor code keeps working.

## History

- 0.2.0 (2026-10-21): streaming (`Delta`, `delta` events, `on_delta`), cancellation (`Runtime.stop`,
  `ProviderCancelled`, stop `cancelled`), schema validation of tool arguments (`schema.py`,
  `tool_result.outcome`, stop `invalid_tool_calls`), structured output (`ModelRequest.output_schema`,
  `AgentSpec.output_schema`, `RunResult.structured`, stop `invalid_output`). Events schema v2.
- 0.1.0: first cut.

## Stability phases

- **0.x (now): settling.** Moving things between core and governor is expected.
  Do it in core-change PRs, one move per PR, with the reason in the description.
- **1.0: frozen.** Once Governor has been stable for a while, cut 1.0. After
  that, breaking changes need a major version bump and a migration note.
