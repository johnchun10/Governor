"""Provider sign-in: one ``/login`` flow per provider, and a status line for each.

API-key providers (Anthropic, OpenAI) ask for a key and save it with
``config.save_credentials``. AWS providers (Bedrock, Mantle) run the AWS CLI's
own login (``aws login`` or ``aws sso login``) or save a Bedrock API key, plus
the region. The flows talk to the user only through ``Prompter``, so the TUI
(or a test) supplies the actual prompts.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import save_credentials, saved_credentials
from .providers import reset_providers

API_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY", "https://platform.claude.com/settings/keys"),
    "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
}
AWS_PROVIDERS = ("bedrock", "mantle")
AWS_KEYS = ("AWS_PROFILE", "AWS_REGION", "AWS_BEARER_TOKEN_BEDROCK")


class Prompter(Protocol):
    def say(self, text: str) -> None: ...
    def text(self, message: str, default: str = "") -> str: ...
    def secret(self, message: str) -> str: ...
    def choose(self, message: str, options: list[tuple[str, str]]) -> str | None: ...
    def run_interactive(self, command: list[str]) -> int:
        """Run an interactive command (e.g. ``aws login``) with the real terminal. Returns its exit code."""
        ...


@dataclass(frozen=True)
class Status:
    ok: bool
    detail: str


def _mask(value: str) -> str:
    return f"…{value[-4:]}" if len(value) > 8 else "set"


def _source(key: str) -> str:
    return "saved" if saved_credentials().get(key) == os.environ.get(key) else "environment"


def status(provider: str) -> Status:
    """Whether credentials for ``provider`` are present. Cheap and offline; `/doctor` makes real calls."""
    if provider in API_KEYS:
        key, _ = API_KEYS[provider]
        if value := os.environ.get(key):
            return Status(True, f"{key} {_mask(value)} ({_source(key)})")
        if provider == "anthropic" and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return Status(True, "ANTHROPIC_AUTH_TOKEN (environment)")
        if provider == "anthropic" and (Path.home() / ".config" / "anthropic").is_dir():
            return Status(True, "`ant auth login` profile")
        return Status(False, "not signed in")
    return _aws_status()


def _aws_status() -> Status:
    from .providers.bedrock import aws_session

    region = os.environ.get("GOVERNOR_AWS_REGION") or os.environ.get("AWS_REGION")
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return Status(True, f"Bedrock API key ({_source('AWS_BEARER_TOKEN_BEDROCK')}), region {region or 'us-east-1'}")
    try:
        session = aws_session()
        creds = session.get_credentials()
    except Exception as exc:
        return Status(False, f"AWS config error: {exc}")
    if creds is None:
        return Status(False, "not signed in")
    return Status(True, f"AWS profile {session.profile_name or 'default'}, region {session.region_name}")


def login(provider: str, ui: Prompter) -> bool:
    """Run the sign-in flow. Returns True when credentials were saved or refreshed."""
    ok = _login_api_key(provider, ui) if provider in API_KEYS else _login_aws(ui)
    if ok:
        reset_providers()  # cached clients hold the old credentials
    return ok


def logout(provider: str) -> list[str]:
    """Forget saved credentials for ``provider``. Returns the keys removed."""
    keys = [API_KEYS[provider][0]] if provider in API_KEYS else list(AWS_KEYS)
    saved = saved_credentials()
    removed = [k for k in keys if k in saved]
    if removed:
        save_credentials(dict.fromkeys(removed))
        reset_providers()
    return removed


def _login_api_key(provider: str, ui: Prompter) -> bool:
    key, url = API_KEYS[provider]
    ui.say(f"Create a key at {url}")
    value = ui.secret(f"{key}: ").strip()
    if not value:
        ui.say("Cancelled.")
        return False
    path = save_credentials({key: value})
    ui.say(f"Saved to {path}")
    return True


def _login_aws(ui: Prompter) -> bool:
    method = ui.choose(
        "How do you sign in to AWS?",
        [
            ("login", "AWS console sign-in (runs `aws login`)"),
            ("sso", "SSO profile (runs `aws sso login`)"),
            ("key", "Bedrock API key"),
            ("existing", "Keep the credentials already configured, just set the region"),
        ],
    )
    if method is None:
        return False
    region = ui.text("AWS region: ", default=os.environ.get("AWS_REGION", "us-east-1")).strip() or "us-east-1"
    values: dict[str, str | None] = {"AWS_REGION": region}

    if method == "key":
        token = ui.secret("Bedrock API key: ").strip()
        if not token:
            ui.say("Cancelled.")
            return False
        values["AWS_BEARER_TOKEN_BEDROCK"] = token
    elif method in ("login", "sso"):
        if shutil.which("aws") is None:
            ui.say("The AWS CLI isn't installed. Install it from https://aws.amazon.com/cli/, then retry.")
            return False
        values["AWS_BEARER_TOKEN_BEDROCK"] = None  # a saved key would take precedence over the login
        if method == "sso":
            profile = ui.text("SSO profile name: ", default=os.environ.get("AWS_PROFILE", "")).strip()
            if not profile:
                ui.say("Cancelled.")
                return False
            command = ["aws", "sso", "login", "--profile", profile]
            values["AWS_PROFILE"] = profile
        else:
            command = ["aws", "login", "--region", region]
            values["AWS_PROFILE"] = None
        ui.say(f"Running: {' '.join(command)}")
        if ui.run_interactive(command) != 0:
            ui.say("AWS sign-in failed; nothing was saved.")
            return False
    path = save_credentials(values)
    ui.say(f"Saved to {path}")
    return True
