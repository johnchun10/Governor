"""Agent profiles: markdown files with YAML frontmatter.

    ---
    name: explore
    description: Read-only codebase exploration. Returns a concise report.
    model: subagent            # main | subagent | a model of the active provider | provider/model
    tools: [read_file, list_files, grep]
    max_turns: 30
    subagent: true             # may be launched through the task tool
    permissions: {bash: deny}  # per-profile overrides of tool defaults
    params: {}                 # provider passthrough (e.g. Converse additionalModelRequestFields)
    output_schema: {type: object, ...}  # optional: the final reply must be JSON matching this
    max_invalid_calls: 3       # consecutive bad calls to one tool that end the run
    ---
    The body is the system prompt.

Built-in profiles ship in ``governor/agents/``. A file with the same name in
``<project>/.governor/agents/`` replaces the built-in.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from agent_core import AgentSpec

BUILTIN_DIR = Path(__file__).parent / "agents"


def _shape(value: Any, hint: Any) -> Any:
    """Shape YAML values by annotation, preserving nullable values and scalar lists."""
    origin, args = get_origin(hint), get_args(hint)
    if hint is Any:
        return value
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        for member in args:
            if member is type(None):
                continue
            try:
                return _shape(value, member)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"expected {hint}")
    if origin is Literal:
        if value not in args:
            raise ValueError(f"expected one of {args}")
        return value
    if origin is dict:
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("expected a mapping")
        key_hint, value_hint = args
        return {_shape(k, key_hint): _shape(v, value_hint) for k, v in value.items()}
    if origin is list:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("expected a list or string")
        return [_shape(item, args[0]) for item in value]
    if value is None:
        raise ValueError(f"expected {hint}, not null")
    if hint is bool and isinstance(value, str):
        if value.lower() not in ("true", "false"):
            raise ValueError("expected true or false")
        return value.lower() == "true"
    return hint(value)


def parse_profile(text: str, source: str = "<string>") -> AgentSpec:
    if not text.startswith("---"):
        raise ValueError(f"{source}: profile must start with YAML frontmatter (---)")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front)
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError(f"{source}: frontmatter must be a mapping")
    if "system_prompt" in meta:
        raise ValueError(f"{source}: system_prompt is the text after the frontmatter, not a frontmatter key")
    names = {field.name for field in fields(AgentSpec) if field.init}
    unknown = meta.keys() - names
    if unknown:
        raise ValueError(f"{source}: unknown frontmatter keys: {', '.join(sorted(map(str, unknown)))}")
    if meta.get("name") is None:
        raise ValueError(f"{source}: frontmatter needs a name")
    hints = get_type_hints(AgentSpec)
    values = {}
    for key, value in meta.items():
        if value is None and hints[key] is str:
            continue  # a bare `description:` means "not set": the field keeps its default, as it always has
        try:
            values[key] = _shape(value, hints[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: invalid {key}: {exc}") from exc
    values["system_prompt"] = body.strip()
    return AgentSpec(**values)


def load_profiles(project_dir: Path | None = None) -> dict[str, AgentSpec]:
    dirs = [BUILTIN_DIR] + ([project_dir / ".governor" / "agents"] if project_dir else [])
    profiles: dict[str, AgentSpec] = {}
    for d in dirs:
        for path in sorted(d.glob("*.md")) if d.is_dir() else []:
            spec = parse_profile(path.read_text(), str(path))
            profiles[spec.name] = spec
    return profiles
