"""Reasoning effort: one scale for every provider, chosen with ``/effort`` or a key.

Governor's levels are ``low``, ``medium``, ``high`` and ``max``; no level leaves
each model at its own default. Each provider spells a level its own way (Claude:
``output_config.effort``; OpenAI's Responses API: ``reasoning.effort``, where
``max`` is ``xhigh``; OpenAI models on Bedrock: ``reasoning_effort``), and a
model with no such setting gets nothing. The level is added at call time, by
wrapping the provider, so every agent in a run tree, subagents and /qa included,
follows the setting as it is when the call is made. An agent profile's own
``params`` are kept, with the chosen effort merged over them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from agent_core import DeltaCallback, ModelInfo, ModelRequest, ModelResponse, Provider

LEVELS = ("low", "medium", "high", "max")

Level = Callable[[], str | None]
"""Reads the current level when a call is made; None means the model's default."""


def params_for(level: str, info: ModelInfo) -> dict[str, Any]:
    """The request fields that set ``level`` on this model, or {} when it has no effort setting."""
    if level not in LEVELS:
        raise ValueError(f"unknown effort {level!r}; use one of {', '.join(LEVELS)}")
    model = info.model_id.lower()
    if info.provider in ("anthropic", "mantle") or "claude" in model:
        return {"output_config": {"effort": level}}
    if info.provider == "openai":
        return {"reasoning": {"effort": "xhigh" if level == "max" else level}}
    if info.provider == "bedrock" and ("openai." in model or "gpt-" in model):
        return {"reasoning_effort": "high" if level == "max" else level}
    return {}


def merged(params: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """``params`` with ``extra`` over it; a dict value (``output_config``) is merged one level deep."""
    out = dict(params)
    for key, value in extra.items():
        current = out.get(key)
        out[key] = {**current, **value} if isinstance(value, dict) and isinstance(current, dict) else value
    return out


class EffortProvider:
    """A provider with the current effort level added to every request it forwards."""

    def __init__(self, inner: Provider, info: ModelInfo, level: Level) -> None:
        self.inner, self.info, self.level = inner, info, level
        self.name = inner.name

    def complete(
        self,
        request: ModelRequest,
        *,
        on_delta: DeltaCallback | None = None,
        stop: threading.Event | None = None,
    ) -> ModelResponse:
        level = self.level()
        extra = params_for(level, self.info) if level else {}
        if extra:
            request = replace(request, params=merged(request.params, extra))
        return self.inner.complete(request, on_delta=on_delta, stop=stop)


def wrap(provider_for: Callable[[ModelInfo], Provider], level: Level) -> Callable[[ModelInfo], Provider]:
    """``provider_for`` whose providers apply the current effort level to every call."""
    return lambda info: EffortProvider(provider_for(info), info, level)
