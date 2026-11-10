"""/effort: one scale, spelled per provider, added to every call by wrapping the provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core import Agent, EventBus, Message, ModelInfo, ModelRequest, StaticPolicy
from agent_core.testing import ScriptedProvider, text
from governor import build_runtime
from governor.config import Settings
from governor.effort import EffortProvider, merged, params_for, wrap

pytestmark = pytest.mark.usefixtures("isolated")


def info(provider: str, model_id: str) -> ModelInfo:
    return ModelInfo(f"{provider}/x", provider, model_id)


@pytest.mark.parametrize(
    ("provider", "model_id", "level", "expected"),
    [
        ("anthropic", "claude-opus-5-5", "high", {"output_config": {"effort": "high"}}),
        ("mantle", "anthropic.claude-opus-5-5", "max", {"output_config": {"effort": "max"}}),
        ("bedrock", "us.anthropic.claude-sonnet-5-5", "low", {"output_config": {"effort": "low"}}),
        ("openai", "gpt-6-astra", "medium", {"reasoning": {"effort": "medium"}}),
        ("openai", "gpt-6-astra", "max", {"reasoning": {"effort": "xhigh"}}),
        ("bedrock", "openai.gpt-oss-120b-1:0", "low", {"reasoning_effort": "low"}),
        ("bedrock", "us.openai.gpt-6-astra", "max", {"reasoning_effort": "high"}),
        ("bedrock", "us.amazon.nova-pro-v1:0", "high", {}),
    ],
)
def test_each_provider_spells_the_level_its_own_way(provider, model_id, level, expected):
    assert params_for(level, info(provider, model_id)) == expected


def test_unknown_level_is_an_error():
    with pytest.raises(ValueError, match="unknown effort"):
        params_for("extreme", info("openai", "gpt-6-astra"))


def test_merged_keeps_the_rest_of_output_config():
    base = {"output_config": {"format": {"type": "json_schema"}, "effort": "low"}, "betas": ["x"]}
    assert merged(base, {"output_config": {"effort": "high"}}) == {
        "output_config": {"format": {"type": "json_schema"}, "effort": "high"},
        "betas": ["x"],
    }
    assert merged({"reasoning": "odd"}, {"reasoning": {"effort": "low"}}) == {"reasoning": {"effort": "low"}}


def test_provider_wrapper_applies_the_level_as_it_is_at_call_time():
    inner = ScriptedProvider([text("one"), text("two")])
    level: list[str | None] = ["high"]
    provider = EffortProvider(inner, info("anthropic", "claude-opus-5-5"), lambda: level[0])
    request = ModelRequest("claude-opus-5-5", (Message.user("hi"),), params={"output_config": {"format": "f"}})
    seen = []

    provider.complete(request, on_delta=seen.append)
    assert inner.requests[0].params == {"output_config": {"format": "f", "effort": "high"}}
    assert inner.requests[0].messages == request.messages and seen  # the rest, and streaming, pass through
    level[0] = None
    provider.complete(request)
    assert inner.requests[1] is request  # untouched without a level
    assert provider.name == inner.name


def test_build_runtime_applies_the_saved_effort_and_the_env_override(isolated: Path, monkeypatch):
    Settings(provider="openai", effort="high").save()
    inner = ScriptedProvider([text("one"), text("two")])
    runtime = build_runtime(
        isolated,
        provider_for=lambda info: inner,
        resolve_model=lambda name: info("openai", "gpt-6-astra"),
        permissions=StaticPolicy(yolo=True),
        events=EventBus([]),
        context_providers=[],
    )
    Agent(runtime.profiles["general"], runtime, cwd=isolated).run("hi")
    assert inner.requests[0].params == {"reasoning": {"effort": "high"}}
    monkeypatch.setenv("GOVERNOR_EFFORT", "low")
    Agent(runtime.profiles["general"], runtime, cwd=isolated).run("hi")
    assert inner.requests[1].params == {"reasoning": {"effort": "low"}}


def test_settings_drop_an_unknown_effort(monkeypatch):
    Settings(provider="openai", effort="silly").save()
    assert Settings.load().effort is None
    settings = Settings(provider="openai", effort="max")
    assert settings.effort_level() == "max"
    monkeypatch.setenv("GOVERNOR_EFFORT", "nonsense")
    assert settings.effort_level() is None


def test_wrap_keeps_the_provider_lookup():
    inner = ScriptedProvider()
    provider_for = wrap(lambda info: inner, lambda: "low")
    provider = provider_for(info("openai", "gpt-6-astra"))
    assert isinstance(provider, EffortProvider) and provider.inner is inner
