"""Slash commands: functions of ``(session, arg)`` that speak through the session's ``io``.

A command is data (``Command``), so any surface can list, complete and run it:
the TUI's suggestions and ``/help`` read the session's table and
``Session.submit`` dispatches to it. The harness's own controls (provider,
model, effort, sign-in, agent, doctor, and the conversation and cost commands)
live here. A feature's command lives in the feature's module, and a mode's
sub-commands in its ``Mode.commands``, where they shadow these while it is on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_core import ModelInfo

from . import auth
from .doctor import doctor, list_inference_profiles
from .effort import params_for
from .models import PROVIDER_LABELS, PROVIDERS, ROLES, catalog, resolve, short_name
from .ports import Tab
from .session import summary

if TYPE_CHECKING:
    from .session import Session


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    run: Callable[[Session, str], None]
    args: Callable[[Session], list[str]] | None = None
    """Completions for the argument."""
    echo: bool = True
    """Show the typed line before running; off for a command that shows its input its own way (``/qa why?``)."""


EFFORT_CHOICES = [
    ("default", "default   the model's own default (or what the agent profile sets)"),
    ("low", "low       quickest and cheapest; for small, clear tasks"),
    ("medium", "medium    some thinking before each reply"),
    ("high", "high      thinks longer; for hard or multi-step work"),
    ("max", "max       thinks as long as it can; slowest and costs the most"),
]
"""The /effort picker, in the order Shift-Tab cycles through; Ctrl-0…4 jump to one."""
EFFORT_LEVELS = tuple(value for value, _ in EFFORT_CHOICES)

SAME = "same"
"""The Q&A model choice that means "whatever model the conversation used"."""
ROLE_TABS = (("main", "Main"), ("subagent", "Subagent"), ("qa", "Q&A"))
"""The pages of the /model picker, in ←/→ order."""


# --- models ----------------------------------------------------------------------------


def _price(info: ModelInfo) -> str:
    if not (info.input_per_mtok or info.output_per_mtok):
        return "price not set"
    return f"${info.input_per_mtok:g}/${info.output_per_mtok:g} per Mtok"


def model_names(session: Session) -> list[str]:
    """The active provider's table, for completions."""
    return list(catalog(session.project_dir)[session.settings.selection().provider])


def ask_model_id(session: Session, provider: str) -> str:
    """Type a raw model id; on Bedrock the account's inference profiles complete it."""
    ids: list[str] = []
    if provider == "bedrock":
        try:
            ids = sorted(p["inferenceProfileId"] for p in list_inference_profiles())
        except Exception as exc:
            session.say(f"couldn't list inference profiles ({type(exc).__name__}); type the id", "dim")
    title = f"Model id on {provider}" + (f" ({len(ids)} inference profiles; → completes)" if ids else "")
    return session.input(title, suggestions=ids).strip()


def pick_models(session: Session, start: str = "main") -> tuple[str, str] | None:
    """The one model picker: a page per role (main, subagent, Q&A), ←/→ between them, opened on ``start``.

    Returns ``(role, name)`` where name is a table name, a raw id, or ``SAME`` on the Q&A page; None when cancelled.
    """
    sel = session.settings.selection()
    models = catalog(session.project_dir)[sel.provider]
    width = max(len(n) for n in models) + 2
    table = [(n, f"{n:<{width}} {_price(m):<22} {m.model_id}") for n, m in models.items()]
    other = ("", "Other model id…")
    same = (SAME, "Same as the conversation (reuses its prompt cache)")
    tabs = [
        Tab(
            role,
            label,
            [*([same] if role == "qa" else []), *table, other],
            sel.roles.get(role, SAME if role == "qa" else None),
        )
        for role, label in ROLE_TABS
    ]
    picked = session.io.choose_tabbed(f"Models on {sel.provider}", tabs, start) if session.io is not None else None
    if picked is None:
        return None
    role, name = picked
    if name == "":
        name = ask_model_id(session, sel.provider)
        if not name:
            return None
    return role, name


def set_role_model(session: Session, role: str, name: str) -> ModelInfo | None:
    """Make ``name`` the model for ``role`` in the active provider. Says what was set, or why it wasn't.

    On the Q&A role, ``SAME`` clears the choice (the conversation's model answers) and returns None.
    """
    sel = session.settings.selection()
    if role == "qa" and name == SAME:
        session.settings.set_model("qa", None)
        session.settings.save()
        session.say("qa model: same as the conversation")
        return None
    try:
        info = resolve(name, sel, session.project_dir)
    except KeyError as exc:
        session.say(str(exc).strip("'\""), "error")
        return None
    if info.provider != sel.provider:
        session.say(f"{name} is a {info.provider} model; switch with /provider {info.provider} first", "error")
        return None
    session.settings.set_model(role, short_name(info))
    session.settings.save()
    session.say(f"{role} model: {short_name(info)} ({info.model_id})")
    return info


def set_effort(session: Session, level: str) -> str:
    """Save the level ("default" clears it). Returns the status line to show."""
    session.settings.effort = None if level == EFFORT_LEVELS[0] else level
    session.settings.save()
    note = ""
    if session.settings.effort:
        try:
            main = resolve("main", session.settings.selection(), session.project_dir)
        except KeyError:
            main = None
        if main is not None and not params_for(level, main):
            note = f" ({short_name(main)} has no effort setting, so it's ignored there)"
    return f"effort: {level}{note}"


# --- the commands ----------------------------------------------------------------------


def provider(session: Session, arg: str) -> None:
    name = arg if arg in PROVIDERS else None
    if name is None:
        options = []
        for p in PROVIDERS:
            mark = "✓ signed in" if auth.status(p).ok else ""
            options.append((p, f"{p:<10} {PROVIDER_LABELS[p]:<44} {mark}"))
        name = session.choose("Model provider", options, default=session.settings.selection().provider)
        if name is None:
            return
    session.settings.provider = name
    session.settings.save()
    sel = session.settings.selection()
    session.say(f"provider: {name} · main: {sel.roles['main']} · subagent: {sel.roles['subagent']}")
    if not auth.status(name).ok:
        session.offer_sign_in(name)


def model(session: Session, arg: str) -> None:
    """``/model``: the picker for all three roles; ``/model <name>`` sets the main model."""
    picked = ("main", arg) if arg else pick_models(session)
    if picked is not None:
        set_role_model(session, *picked)


def effort(session: Session, arg: str) -> None:
    level = arg or None
    if level is None:
        default = session.settings.effort or EFFORT_LEVELS[0]
        level = session.choose("Effort: how hard the model thinks", EFFORT_CHOICES, default=default)
        if level is None:
            return
    elif level not in EFFORT_LEVELS:
        session.say(f"unknown effort {level!r}; use one of {', '.join(EFFORT_LEVELS)}", "error")
        return
    session.say(set_effort(session, level))


def login(session: Session, arg: str) -> None:
    name = arg if arg in PROVIDERS else None
    if name is None:
        options = []
        for p in PROVIDERS:
            state = auth.status(p)
            options.append((p, f"{p:<10} {'✓' if state.ok else '✗'} {state.detail}"))
        name = session.choose("Sign in to", options, default=session.settings.selection().provider)
        if name is None:
            return
    session.sign_in(name)


def logout(session: Session, arg: str) -> None:
    name = arg if arg in PROVIDERS else session.settings.selection().provider
    removed = auth.logout(name)
    session.say(f"removed {', '.join(removed)}" if removed else f"no saved credentials for {name}")
    if (state := auth.status(name)).ok:
        session.say(f"still signed in through the environment: {state.detail}", "dim")


def agent(session: Session, arg: str) -> None:
    profiles = session.runtime.profiles
    name = arg if arg in profiles else None
    if name is None:
        options = [(n, f"{n:<10} {p.description}") for n, p in sorted(profiles.items())]
        name = session.choose("Agent", options, default=session.settings.agent)
        if name is None:
            return
    session.settings.agent = name
    session.settings.save()
    session.say(f"agent: {name}")


def doctor_command(session: Session, _: str) -> None:
    sel = session.settings.selection()
    try:
        models = {info.alias: info for info in (resolve(role, sel, session.project_dir) for role in ROLES)}
    except KeyError as exc:
        session.say(str(exc).strip("'\""), "error")
        return
    doctor(
        sel.provider, list(models.values()), out=lambda line: session.say(line, "error" if line[:1] == "✗" else "plain")
    )


def clear(session: Session, _: str) -> None:
    session.clear()
    if session.io is not None:
        session.io.clear()
    session.say("new conversation", "dim")


def cost(session: Session, _: str) -> None:
    session.say(f"session: {summary(session.total)}")


def yolo(session: Session, _: str) -> None:
    permissions = session.runtime.permissions
    permissions.yolo = not permissions.yolo  # type: ignore[attr-defined]
    session.say("yolo on: every tool call is approved" if permissions.yolo else "yolo off")  # type: ignore[attr-defined]


BUILTIN: tuple[Command, ...] = (
    Command("provider", "choose the model provider", provider, lambda s: list(PROVIDERS)),
    Command("model", "choose the main, subagent and Q&A models (from this provider)", model, model_names),
    Command(
        "effort", "choose how hard the model thinks (low, medium, high, max)", effort, lambda s: list(EFFORT_LEVELS)
    ),
    Command("login", "sign in to a provider", login, lambda s: list(PROVIDERS)),
    Command("logout", "forget a provider's saved credentials", logout, lambda s: list(PROVIDERS)),
    Command("agent", "choose the agent profile", agent, lambda s: sorted(s.runtime.profiles)),
    Command("doctor", "check sign-in and test the current models", doctor_command),
    Command("clear", "start a new conversation", clear),
    Command("cost", "show this session's token use and cost", cost),
    Command("yolo", "toggle approving every tool call without asking", yolo),
)
"""The harness's own commands. Features add theirs through ``features.py``; a surface adds its own (``exit``)."""
