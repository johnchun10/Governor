"""Per-provider model selection, saved settings and credentials, login flows, and the `govern` entry point."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from governor import auth
from governor.cli import main
from governor.config import Settings, load_environment, save_credentials, saved_credentials
from governor.models import Selection, catalog, resolve

pytestmark = pytest.mark.usefixtures("isolated")


# --- models ----------------------------------------------------------------------


def test_roles_and_names_resolve_within_the_active_provider():
    openai = Selection.default("openai")
    assert resolve("main", openai).model_id == "gpt-6-astra"
    assert resolve(None, openai).model_id == "gpt-6-astra"
    assert resolve("subagent", openai).model_id == "gpt-6-luna"
    assert resolve("fast", openai).model_id == "gpt-6-luna"  # the role's old name still resolves
    assert resolve("sol", openai).alias == "openai/sol"
    assert resolve("gpt-6.1-sol", openai).alias == "openai/sol"  # by model id
    assert resolve("terra", openai).model_id == "gpt-6.1-terra"

    bedrock = Selection.default("bedrock")
    assert resolve("main", bedrock).model_id == "us.anthropic.claude-opus-5-5"
    assert resolve("anthropic/haiku", bedrock).provider == "anthropic"  # explicit provider wins
    raw = resolve("us.meta.llama4-maverick-17b-instruct-v1:0", bedrock)
    assert (raw.provider, raw.input_per_mtok) == ("bedrock", 0)
    assert resolve("claude-opus-4-8", Selection.default("anthropic")).model_id == "claude-opus-4-8"
    with pytest.raises(KeyError, match="known: astra, sol, terra, luna"):
        resolve("opus", openai)  # Claude names don't exist on OpenAI


def test_models_file_adds_and_overrides_per_provider(isolated: Path):
    (isolated / "models.toml").write_text(
        '[bedrock.qwen]\nmodel_id = "qwen.qwen3"\ninput_per_mtok = 0.22\n\n'
        '[bedrock.opus]\nmodel_id = "global.anthropic.claude-opus-5-5"\n'
    )
    bedrock = Selection.default("bedrock")
    assert resolve("qwen", bedrock).input_per_mtok == 0.22
    opus = resolve("opus", bedrock)
    assert opus.model_id == "global.anthropic.claude-opus-5-5"
    assert opus.input_per_mtok == 4.0  # untouched fields keep the built-in value


def test_models_file_is_read_from_the_project_dir(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOVERNOR_MODELS_FILE")
    project = isolated / "elsewhere"
    (project / ".governor").mkdir(parents=True)
    (project / ".governor" / "models.toml").write_text('[openai.mini]\nmodel_id = "gpt-6-mini"\n')
    openai = Selection.default("openai")
    assert "mini" in catalog(project)["openai"]
    assert resolve("mini", openai, project).model_id == "gpt-6-mini"
    with pytest.raises(KeyError):
        resolve("mini", openai)  # nothing is read relative to the process cwd


# --- settings and credentials ----------------------------------------------------


def test_model_choices_are_remembered_per_provider():
    settings = Settings.load()
    assert settings.provider is None
    settings.provider = "openai"
    settings.set_model("main", "sol")
    settings.provider = "anthropic"
    settings.set_model("subagent", "sonnet")
    settings.save()

    loaded = Settings.load()
    assert loaded.selection().roles == {"main": "opus", "subagent": "sonnet"}
    loaded.provider = "openai"
    assert loaded.selection().roles == {"main": "sol", "subagent": "luna"}
    loaded.set_model("main", "astra")  # back to the default: not stored
    assert "openai" not in loaded.models


def test_settings_saved_before_the_rename_still_load(isolated: Path):
    (isolated / "home").mkdir()
    (isolated / "home" / "settings.json").write_text('{"provider": "openai", "models": {"openai": {"fast": "sol"}}}')
    assert Settings.load().selection().roles == {"main": "astra", "subagent": "sol"}


def test_env_overrides_selection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOVERNOR_PROVIDER", "openai")
    monkeypatch.setenv("GOVERNOR_MODEL", "luna")
    sel = Settings(provider="anthropic").selection()
    assert (sel.provider, sel.roles["main"]) == ("openai", "luna")


def test_credentials_file_is_private_and_lowest_precedence(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    path = save_credentials({"OPENAI_API_KEY": "sk-saved", "ANTHROPIC_API_KEY": "sk-ant"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert os.environ["OPENAI_API_KEY"] == "sk-saved"
    save_credentials({"ANTHROPIC_API_KEY": None})
    assert saved_credentials() == {"OPENAI_API_KEY": "sk-saved"}
    assert "ANTHROPIC_API_KEY" not in os.environ

    monkeypatch.delenv("OPENAI_API_KEY")
    (isolated / ".env").write_text("OPENAI_API_KEY=sk-dotenv\n")
    load_environment(isolated)
    assert os.environ["OPENAI_API_KEY"] == "sk-dotenv"  # project .env beats saved credentials


# --- login -----------------------------------------------------------------------


class FakeUI:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)

    def text(self, message: str, default: str = "") -> str:
        return self.answers.pop(0) or default

    secret = text

    def choose(self, message: str, options: list[tuple[str, str]], default: str | None = None) -> str | None:
        answer = self.answers.pop(0)
        assert answer in [value for value, _ in options]
        return answer


def test_api_key_login_and_logout():
    assert not auth.status("openai").ok
    assert auth.login("openai", FakeUI(["sk-test-1234567890"]))
    state = auth.status("openai")
    assert state.ok and "…7890 (saved)" in state.detail
    assert auth.logout("openai") == ["OPENAI_API_KEY"]
    assert not auth.status("openai").ok
    assert not auth.login("anthropic", FakeUI([""]))  # empty key cancels


def test_aws_login_with_bedrock_api_key():
    assert auth.login("bedrock", FakeUI(["key", "eu-west-1", "bedrock-key"]))
    assert saved_credentials() == {"AWS_BEARER_TOKEN_BEDROCK": "bedrock-key", "AWS_REGION": "eu-west-1"}
    state = auth.status("mantle")
    assert state.ok and "eu-west-1" in state.detail


# --- CLI -------------------------------------------------------------------------


def test_cli_is_the_tui_and_nothing_else(capsys: pytest.CaptureFixture[str]):
    assert main([]) == 2  # no terminal under pytest
    assert "needs a terminal" in capsys.readouterr().err
    assert main(["run", "say hi"]) == 2 and "usage: govern" in capsys.readouterr().err
    assert main(["--help"]) == 0 and "usage: govern" in capsys.readouterr().out
