"""Model catalog, organized by provider.

Each provider has its own table of short model names (``opus``, ``astra``...)
and two default roles: ``main`` (the model you talk to) and ``subagent`` (cheap
subagent work such as ``explore``). A user picks a provider, then a model from
that provider's table, and the choice is remembered per provider (see
``config.py``).

Agent profiles name a role (``model: main`` / ``model: subagent``), a model in the
active provider (``model: opus``), or an explicit ``provider/model``. Anything
else is treated as a raw model id on the active provider, priced at $0.

The project's ``.governor/models.toml`` (or ``$GOVERNOR_MODELS_FILE``) adds or
overrides entries, one table per provider. ``catalog`` and ``resolve`` take the
project directory; nothing is read relative to the process cwd::

    [bedrock.qwen]
    model_id = "qwen.qwen3-coder-480b-a35b-v1:0"
    input_per_mtok = 0.22
    output_per_mtok = 1.80
    supports_cache = false

Prices are USD per million tokens. Claude prices are Anthropic's list prices
and OpenAI prices are OpenAI's (standard tier, prompts up to 272K input tokens,
from https://developers.openai.com/api/docs/pricing.md on 2026-10-17; terra on
2026-10-29). Bedrock bills at AWS's own rates, so treat Bedrock costs as estimates.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from agent_core import ModelInfo

from .providers import PROVIDERS

PROVIDER_LABELS = {
    "bedrock": "Amazon Bedrock, every model family (AWS)",
    "mantle": "Amazon Bedrock, Claude Messages API (AWS)",
    "anthropic": "Anthropic API (API key)",
    "openai": "OpenAI API (API key)",
}

ROLES = ("main", "subagent")


def _claude(provider: str, name: str, model_id: str, inp: float, out: float, read: float) -> ModelInfo:
    # Cache writes (5-minute TTL) are billed at 1.25x the input price.
    return ModelInfo(f"{provider}/{name}", provider, model_id, inp, out, read, inp * 1.25)


def _openai(provider: str, name: str, model_id: str, *prices: float, cache: bool = True) -> ModelInfo:
    return ModelInfo(f"{provider}/{name}", provider, model_id, *prices, supports_cache=cache)


BUILTIN: dict[str, list[ModelInfo]] = {
    "bedrock": [
        # Cross-region inference profiles; `/doctor` checks they exist in your region.
        _claude("bedrock", "fable", "us.anthropic.claude-fable-5-1", 10.0, 50.0, 0.25),
        _claude("bedrock", "opus", "us.anthropic.claude-opus-5-5", 4.0, 20.0, 0.20),
        _claude("bedrock", "sonnet", "us.anthropic.claude-sonnet-5-5", 2.0, 10.0, 0.20),
        _claude("bedrock", "haiku", "us.anthropic.claude-haiku-5-5", 0.10, 0.50, 0.01),
        # OpenAI frontier models on Bedrock are separate Marketplace offers, so the AWS Pricing API
        # has no prices for them: set real ones in models.toml.
        _openai("bedrock", "astra", "us.openai.gpt-6-astra", cache=False),
        _openai("bedrock", "sol", "us.openai.gpt-6.1-sol", cache=False),
        _openai("bedrock", "luna", "us.openai.gpt-6-luna", cache=False),
        # Open-weight and Amazon models (on-demand, prices from the AWS Pricing API).
        _openai("bedrock", "gpt-oss", "openai.gpt-oss-120b-1:0", 0.15, 0.60, cache=False),
        _openai("bedrock", "gpt-oss-small", "openai.gpt-oss-20b-1:0", 0.07, 0.30, cache=False),
        ModelInfo("bedrock/nova-pro", "bedrock", "us.amazon.nova-pro-v1:0", 0.80, 3.20, 0.20, 0.0),
    ],
    "mantle": [
        _claude("mantle", "opus", "anthropic.claude-opus-5-5", 4.0, 20.0, 0.20),
        _claude("mantle", "sonnet", "anthropic.claude-sonnet-5-5", 2.0, 10.0, 0.20),
    ],
    "anthropic": [
        _claude("anthropic", "fable", "claude-fable-5-1", 10.0, 50.0, 0.25),
        _claude("anthropic", "opus", "claude-opus-5-5", 4.0, 20.0, 0.20),
        _claude("anthropic", "sonnet", "claude-sonnet-5-5", 2.0, 10.0, 0.20),
        _claude("anthropic", "haiku", "claude-haiku-5-5", 0.10, 0.50, 0.01),
    ],
    "openai": [
        _openai("openai", "astra", "gpt-6-astra", 10.0, 50.0, 1.0, 12.50),
        _openai("openai", "sol", "gpt-6.1-sol", 2.0, 10.0, 0.10, 2.50),
        _openai("openai", "terra", "gpt-6.1-terra", 1.0, 5.0, 0.05, 1.25),
        _openai("openai", "luna", "gpt-6-luna", 0.10, 0.50, 0.01, 0.125),
    ],
}

DEFAULT_ROLES: dict[str, dict[str, str]] = {
    "bedrock": {"main": "opus", "subagent": "haiku"},
    "mantle": {"main": "opus", "subagent": "sonnet"},
    "anthropic": {"main": "opus", "subagent": "haiku"},
    "openai": {"main": "astra", "subagent": "luna"},
}


def short_name(info: ModelInfo) -> str:
    """bedrock/opus -> opus (raw ids have no table name and come back unchanged)."""
    return info.alias.split("/", 1)[1] if "/" in info.alias else info.alias


def _models_file(project_dir: Path | None) -> Path | None:
    """``$GOVERNOR_MODELS_FILE``, else the project's ``.governor/models.toml``; None without a project."""
    if env := os.environ.get("GOVERNOR_MODELS_FILE"):
        return Path(env)
    return project_dir / ".governor" / "models.toml" if project_dir is not None else None


def _load_file(project_dir: Path | None) -> dict[str, dict[str, ModelInfo]]:
    path = _models_file(project_dir)
    if path is None or not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    out: dict[str, dict[str, ModelInfo]] = {}
    for provider, entries in data.items():
        if provider not in PROVIDERS:
            raise ValueError(f"{path}: unknown provider table [{provider}]; use one of {', '.join(PROVIDERS)}")
        builtin = {short_name(m): m for m in BUILTIN[provider]}
        for name, entry in entries.items():
            base = builtin.get(name)
            alias = f"{provider}/{name}"
            if base is not None:
                out.setdefault(provider, {})[name] = replace(base, **entry)
            else:
                out.setdefault(provider, {})[name] = ModelInfo(alias=alias, provider=provider, **entry)
    return out


def catalog(project_dir: Path | None = None) -> dict[str, dict[str, ModelInfo]]:
    """provider -> model name -> ModelInfo, with the project's models.toml applied."""
    models = {p: {short_name(m): m for m in BUILTIN[p]} for p in PROVIDERS}
    for provider, entries in _load_file(project_dir).items():
        models[provider].update(entries)
    return models


@dataclass
class Selection:
    """Which provider is active and which model fills each role there."""

    provider: str
    roles: dict[str, str]
    """role -> model name (or raw id) in ``provider``."""

    @classmethod
    def default(cls, provider: str) -> Selection:
        return cls(provider, dict(DEFAULT_ROLES[provider]))


def resolve(name: str | None, selection: Selection, project_dir: Path | None = None) -> ModelInfo:
    """Turn a role, model name, ``provider/model`` or raw id into a ModelInfo."""
    models = catalog(project_dir)
    name = name or "main"
    if name == "fast":  # the subagent role's name before 2026-10-30; profiles may still say it
        name = "subagent"
    provider = selection.provider
    if name in ROLES:
        name = selection.roles.get(name) or DEFAULT_ROLES[provider][name]
    if "/" in name and name.split("/", 1)[0] in PROVIDERS:
        provider, name = name.split("/", 1)
    if name in models[provider]:
        return models[provider][name]
    for info in models[provider].values():
        if info.model_id == name:
            return info
    if any(c.isdigit() or c in ".:" for c in name):
        # Looks like a model id (claude-opus-4-8, us.meta.llama...), not a mistyped table name.
        return ModelInfo(alias=f"{provider}/{name}", provider=provider, model_id=name, supports_cache=False)
    raise KeyError(f"unknown {provider} model {name!r}; known: {', '.join(models[provider])}")
