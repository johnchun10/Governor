"""User settings and saved credentials, kept in ``~/.config/governor`` (or ``$GOVERNOR_HOME``).

* ``settings.json``: the active provider, the model chosen for each role (main, subagent, qa) in each
  provider (so switching providers and back restores your picks), the agent, the effort, and each
  feature's own settings under ``features`` (``Settings.of("budget")``).
* ``credentials``: ``KEY=VALUE`` lines written by ``/login`` (mode 600), loaded
  into the environment at startup.

Precedence for credentials, highest first: the shell environment, the
project's ``.env``, then the saved credentials file. For the model selection and
effort, ``GOVERNOR_PROVIDER``, ``GOVERNOR_MODEL`` and ``GOVERNOR_EFFORT`` override
the saved settings.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .effort import LEVELS
from .models import DEFAULT_ROLES, PROVIDERS, Selection


def home() -> Path:
    return Path(os.environ.get("GOVERNOR_HOME") or Path.home() / ".config" / "governor")


# --- env files -------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            out[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return out


def load_environment(project_dir: Path) -> None:
    """Fill os.environ from the project .env, then saved credentials. Never overrides what is set."""
    for path in (project_dir / ".env", home() / "credentials"):
        for key, value in read_env_file(path).items():
            os.environ.setdefault(key, value)


def saved_credentials() -> dict[str, str]:
    return read_env_file(home() / "credentials")


def save_credentials(values: dict[str, str | None]) -> Path:
    """Merge into the credentials file (None removes a key) and apply to this process."""
    path = home() / "credentials"
    current = saved_credentials()
    for key, value in values.items():
        if value is None:
            current.pop(key, None)
            os.environ.pop(key, None)
        else:
            current[key] = value
            os.environ[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("# Written by `govern` /login. Keep private.\n")
        fh.writelines(f"{k}={v}\n" for k, v in sorted(current.items()))
    os.chmod(path, 0o600)
    return path


# --- settings --------------------------------------------------------------------


@dataclass
class Settings:
    provider: str | None = None
    """None until the user picks one (first run)."""
    models: dict[str, dict[str, str]] = field(default_factory=dict)
    """provider -> role -> model name, only for choices that differ from the defaults."""
    agent: str = "general"
    effort: str | None = None
    """How hard the model thinks, one of ``effort.LEVELS``; None leaves each model at its own default."""
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Each feature's own settings under its name (``of("budget")``); a feature never gets a field here."""

    @classmethod
    def load(cls) -> Settings:
        path = home() / "settings.json"
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
        settings = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        for roles in settings.models.values():  # the subagent role was called fast before 2026-10-30
            if "fast" in roles:
                roles.setdefault("subagent", roles.pop("fast"))
        if settings.provider not in PROVIDERS:
            settings.provider = None
        if settings.effort not in LEVELS:
            settings.effort = None
        features = settings.features if isinstance(settings.features, dict) else {}
        # A hand-edited file must not stop govern from starting: a feature's namespace is a dict or nothing.
        settings.features = {k: v for k, v in features.items() if isinstance(k, str) and isinstance(v, dict)}
        return settings

    def save(self) -> None:
        path = home() / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["features"] = {name: values for name, values in self.features.items() if values}
        path.write_text(json.dumps(data, indent=2) + "\n")

    def of(self, feature: str) -> dict[str, Any]:
        """The settings namespace of ``feature``: a dict it owns, saved with the rest. Keys are the feature's."""
        if not isinstance(self.features.get(feature), dict):
            self.features[feature] = {}
        return self.features[feature]

    def selection(self) -> Selection:
        provider = os.environ.get("GOVERNOR_PROVIDER") or self.provider or "bedrock"
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}")
        roles = {**DEFAULT_ROLES[provider], **self.models.get(provider, {})}
        if os.environ.get("GOVERNOR_MODEL"):
            roles["main"] = os.environ["GOVERNOR_MODEL"]
        return Selection(provider, roles)

    def effort_level(self) -> str | None:
        """The effort to apply to model calls; ``GOVERNOR_EFFORT`` overrides the saved choice."""
        level = os.environ.get("GOVERNOR_EFFORT") or self.effort
        return level if level in LEVELS else None

    def set_model(self, role: str, name: str | None) -> None:
        """``None``, or the provider's default for the role, clears the choice."""
        provider = self.selection().provider
        chosen = self.models.setdefault(provider, {})
        if name is None or DEFAULT_ROLES[provider].get(role) == name:
            chosen.pop(role, None)
        else:
            chosen[role] = name
        if not chosen:
            self.models.pop(provider)
