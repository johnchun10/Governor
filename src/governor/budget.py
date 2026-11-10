"""``/budget``: a cap on what this session spends, enforced by a sink.

The cap is a dollar figure kept in the settings (``Settings.of("budget")``).
``BudgetSink`` adds up the cost of every ``model_call`` as it happens, subagents
and Q&A included, and when the total reaches the cap it stops the run the way
Esc does: the model call in flight is not repeated, pending tool calls are
refused, and the turn reports ``budget`` rather than a user stop. A run started
under a spent budget stops before its first model call. The cap is a backstop
after the call that crosses it, not a pre-call limit: it can be overshot by the
calls already under way when it is reached, one in a plain run and one per
subagent when several run in parallel (a precise cap would be a core change). When the call that crosses it was the run's final
reply the finished turn is kept (``Session.run`` counts a feature's stop only
where the loop cut something short). ``/budget`` picks or clears it; the status
bar shows it while one is set.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from agent_core import Event

if TYPE_CHECKING:
    from .session import Session

NAME = "budget"
KEY = "cap_usd"
OFF = "off"
OTHER = ""
MAX_CAP = 1_000_000.0
"""Above this a cap is a typo, not a limit (and would print as a 300-digit figure)."""
CHOICES = [
    (OFF, f"{OFF:<8} no cap"),
    *((f"{cap:g}", f"${cap:<7g} stop a run once this session has spent this much") for cap in (0.5, 1, 2, 5, 10, 25)),
    (OTHER, "Other amount…"),
]
"""The /budget picker. A run stops after the model call that crosses the cap, so calls under way can overshoot it."""


def parse_cap(value: object) -> float | None:
    """A usable cap from user input or the saved settings: a positive, finite number of dollars, else None."""
    if isinstance(value, bool):
        return None
    try:
        cap = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return cap if math.isfinite(cap) and 0 < cap <= MAX_CAP else None


def dollars(amount: float) -> str:
    return f"${amount:.2f}" if amount >= 0.01 else f"${amount:g}"


class BudgetSink:
    """Sums every model call's cost and stops the run at the cap (``session.request_stop("budget")``)."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.spent = 0.0
        """Cost of every model call this session so far, as the events arrive (``session.total`` lags a run)."""
        self.tripped = False
        """The current run was stopped by the cap (reset when a top-level run starts)."""

    @property
    def cap(self) -> float | None:
        """The saved cap, or None when there is none or it is unusable (a hand-edited settings file)."""
        return parse_cap(self.session.settings.of(NAME).get(KEY))

    def handle(self, event: Event) -> None:
        if event.type == "run_start" and event.depth == 0:
            self.tripped = False
        elif event.type == "model_call":
            self.spent += float(event.data["usage"].get("cost_usd", 0.0))
        else:
            return
        cap = self.cap
        if cap is None or self.spent < cap or self.tripped:
            return
        self.tripped = True
        self.session.say(
            f"budget: {dollars(cap)} reached ({dollars(self.spent)} spent); /budget raises or clears it", "dim"
        )
        self.session.request_stop(NAME)


def sinks(session: Session) -> list[BudgetSink]:
    return [BudgetSink(session)]


def status(session: Session) -> list[tuple[str, str]]:
    sink = session.sinks.find(BudgetSink)
    if sink is None or (cap := sink.cap) is None:
        return []
    return [(f"budget: {dollars(cap)} reached", "warning") if sink.spent >= cap else (f"budget: {dollars(cap)}", "")]


def args(session: Session) -> list[str]:
    return [value for value, _ in CHOICES if value != OTHER]


def command(session: Session, arg: str) -> None:
    """``/budget``: pick the cap; ``/budget 2.50`` sets one, ``/budget off`` clears it. Saved across sessions."""
    sink = session.sinks.find(BudgetSink)
    choice = arg.strip().lstrip("$")
    if not choice:
        current = sink.cap if sink is not None else None
        default = (
            OFF if current is None else next((v for v, _ in CHOICES if v and v != OFF and float(v) == current), OTHER)
        )
        choice = session.choose("Budget: stop a run once this session has spent", CHOICES, default=default)
        if choice is None:
            return
        if choice == OTHER:
            choice = session.input("Cap in dollars (e.g. 3.50)").strip().lstrip("$")
            if not choice:
                return
    settings = session.settings.of(NAME)
    if choice.lower() == OFF:
        settings.pop(KEY, None)
        session.settings.save()
        session.say("budget: off")
        return
    cap = parse_cap(choice)
    if cap is None:
        session.say(f"budget: {choice!r} isn't a usable cap; use a positive number of dollars, or off", "error")
        return
    settings[KEY] = cap
    session.settings.save()
    spent = f" ({dollars(sink.spent)} spent so far)" if sink is not None and sink.spent else ""
    session.say(f"budget: {dollars(cap)} for this session{spent}; a run stops after the model call that reaches it")
