"""Q&A about the conversation so far: ``/qa``.

A question branches off the conversation. It is sent as one more user message
after the conversation as it stood when ``/qa`` began, and neither the question
nor its answer is ever added to the conversation. Every question starts from
there again; a follow-up continues the previous question's exchange instead.

The branch reuses the conversation's agent profile, model and tools unchanged,
so its request repeats the conversation's prompt byte for byte and the provider
serves that part from the prompt cache: only the question is new input. That is
why the instructions for answering travel with the question rather than in the
system prompt, and why the tools stay declared (Converse also rejects tool
blocks in the history without them) while every call is refused.

``QAMode`` is the input mode: while it is on, typed lines are questions, ``/model``
picks the answering model and ``/done`` (or Esc) leaves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from agent_core import AgentSpec, Decision, Message, ModelInfo, RunResult, Runtime

from ..commands import SAME, Command, model_names, pick_models, set_role_model
from ..models import resolve, short_name
from ..session import strip_reasoning

if TYPE_CHECKING:
    from ..session import Session

PREAMBLE = """<qa>
The user paused to ask a question about this conversation. Answer from what is already in it, including your \
earlier reasoning and tool results. Don't call tools or continue the task, and if the conversation doesn't \
contain the answer, say so. Be brief: a few sentences or a short list, unless the question asks for detail.
</qa>

"""

PLACEHOLDER = "Ask about the conversation · start with + to follow up · Esc leaves Q&A"
HINT = "Each question starts fresh from the conversation · start a line with + to follow up · Esc or /done leaves"


class NoTools:
    """Permission policy for Q&A: every tool call is refused."""

    def check(
        self, tool: str, args: dict[str, Any], default: Decision, profile: dict[str, Decision] | None = None
    ) -> str | None:
        return "tools are off in Q&A mode; answer from the conversation so far"


@dataclass
class QASession:
    base: tuple[Message, ...]
    """The conversation when /qa began."""
    spec: AgentSpec
    runtime: Runtime
    model: ModelInfo
    thread: list[Message] = field(default_factory=list)
    """The last question and its answer, which a follow-up continues."""

    @classmethod
    def start(
        cls,
        history: Sequence[Message],
        spec: AgentSpec,
        runtime: Runtime,
        model: ModelInfo,
        conversation_model: ModelInfo,
    ) -> QASession:
        """``spec`` and ``conversation_model`` are what produced ``history``; ``model`` answers the questions."""
        base = tuple(history)
        if (model.provider, model.model_id) != (conversation_model.provider, conversation_model.model_id):
            base = tuple(strip_reasoning(base))
        return cls(
            base,
            # A second turn lets the model answer after a refused tool call.
            replace(spec, name="qa", model=model.alias, max_turns=2),
            replace(runtime, permissions=NoTools()),
            model,
        )

    def request(self, question: str, follow_up: bool = False) -> tuple[list[Message], str]:
        """The history and prompt that ask ``question``; ``follow_up`` continues the last exchange."""
        if follow_up and self.thread:
            return [*self.base, *self.thread], question
        return list(self.base), PREAMBLE + question

    def record(self, result: RunResult) -> None:
        """Remember the exchange a follow-up would continue."""
        if result.stop == "error":
            return
        thread = result.messages[len(self.base) :]
        # A follow-up has to come after an answer; a run cut short by tool calls ends on their results.
        self.thread = thread if thread and thread[-1].role == "assistant" else []


class QAMode:
    name = "Q&A"
    placeholder = PLACEHOLDER

    def __init__(self, session: Session) -> None:
        if not session.history or session.conversation is None:
            raise ValueError("/qa answers questions about the conversation, and there isn't one yet")
        self.spec, self.conversation_model = session.conversation
        self.history = list(session.history)
        self.qa = self._start(session, self._answerer(session))
        self.commands = {
            "model": Command(
                "model",
                "choose the model that answers (same = the conversation's)",
                self.pick_model,
                lambda s: [SAME, *model_names(s)],
            ),
            "done": Command("done", "leave Q&A (or Esc)", lambda s, _: s.leave_mode()),
        }

    @property
    def hint(self) -> str:
        lines = [
            f"Q&A about this conversation, answered by {short_name(self.qa.model)}; nothing here is added to it",
            HINT,
        ]
        if caveat := self._caveat():
            lines.append(caveat)
        return "\n".join(lines)

    def _answerer(self, session: Session) -> ModelInfo:
        sel = session.settings.selection()
        return resolve(sel.roles["qa"], sel, session.project_dir) if "qa" in sel.roles else self.conversation_model

    def _start(self, session: Session, answerer: ModelInfo) -> QASession:
        return QASession.start(self.history, self.spec, session.runtime, answerer, self.conversation_model)

    def _caveat(self) -> str | None:
        answerer, model = self.qa.model, self.conversation_model
        if (answerer.provider, answerer.model_id) == (model.provider, model.model_id):
            return None
        return (
            f"{short_name(answerer)} can't reuse {short_name(model)}'s prompt cache or read its reasoning; "
            "/model same switches back"
        )

    def pick_model(self, session: Session, arg: str) -> None:
        """``/model`` in Q&A opens the picker on its Q&A page; ``/model <name>`` sets the model that answers."""
        picked = ("qa", arg) if arg else pick_models(session, start="qa")
        if picked is None:
            return
        role, name = picked
        if role != "qa":  # another page was used: an ordinary model change
            set_role_model(session, role, name)
            return
        if name == SAME:
            set_role_model(session, "qa", SAME)
            answerer = self.conversation_model
        else:
            chosen = set_role_model(session, "qa", name)
            if chosen is None:
                return
            answerer = chosen
        self.qa = self._start(session, answerer)
        if caveat := self._caveat():
            session.say(caveat, "dim")

    def submit(self, session: Session, line: str) -> None:
        """Answer one question. A leading + makes it a follow-up."""
        follow_up = line.startswith("+") and bool(self.qa.thread)
        question = line.removeprefix("+").strip()
        if not question:
            return
        session.say(("+ " if follow_up else "? ") + question, "user")
        history, prompt = self.qa.request(question, follow_up)
        turn = session.run(self.qa.spec, prompt, history=history, runtime=self.qa.runtime)
        if turn.result is not None:
            self.qa.record(turn.result)
        if turn.error is not None:
            session.say(turn.error, "error")
        elif turn.stopped:
            session.say("stopped" if turn.stopped == "user" else f"stopped by {turn.stopped}", "dim")
        elif turn.ran:
            session.say(turn.status(), "dim")

    def leave(self, session: Session) -> None:
        pass

    def title(self, draft: str) -> str | None:
        return "Q&A · follow-up" if self.qa.thread and draft.startswith("+") else "Q&A · new question"


def enter(session: Session, arg: str) -> None:
    """``/qa``: switch the input to questions about the conversation; ``/qa why?`` asks straight away."""
    mode = session.mode
    if not isinstance(mode, QAMode):
        try:
            mode = QAMode(session)
        except (ValueError, KeyError) as exc:
            session.say(str(exc).strip("'\""), "error")
            return
        session.enter_mode(mode)
    elif not arg:
        session.say(f"already in Q&A. {HINT}", "dim")
    if arg:
        mode.submit(session, arg)
