from __future__ import annotations

from pathlib import Path

import pytest

from agent_core import Collector, EventBus, ModelInfo, StaticPolicy, ToolContext
from agent_core.testing import ScriptedProvider
from governor import build_runtime


@pytest.fixture
def runtime(tmp_path: Path):
    """The real Governor runtime (real tools and profiles), with a scripted model."""
    provider = ScriptedProvider()
    collector = Collector()
    rt = build_runtime(
        tmp_path,
        provider_for=lambda info: provider,
        resolve_model=lambda name: ModelInfo(name or "fake", "fake", "fake"),
        permissions=StaticPolicy(yolo=True),
        events=EventBus([collector]),
        context_providers=[],
    )
    return rt, provider, collector


@pytest.fixture
def ctx(runtime, tmp_path: Path) -> ToolContext:
    rt, _, _ = runtime
    return ToolContext(rt, tmp_path, "test", "run", 0)


ENV = [
    "GOVERNOR_PROVIDER",
    "GOVERNOR_MODEL",
    "GOVERNOR_EFFORT",
    "GOVERNOR_AGENT",
    "GOVERNOR_ALLOW",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_BEARER_TOKEN_BEDROCK",
]


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep settings, credentials and the environment away from the real ~/.config/governor."""
    monkeypatch.setenv("GOVERNOR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GOVERNOR_MODELS_FILE", str(tmp_path / "models.toml"))
    monkeypatch.setenv("HOME", str(tmp_path))  # no real ~/.config/anthropic or ~/.aws
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path
