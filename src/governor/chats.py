"""Saved chats: the conversation store behind ``/chats``.

Every completed turn in the TUI is written to ``<project>/.governor/chats/<id>.json``
with the messages exactly as the model saw them (reasoning blocks included), so
a chat can be resumed later with its prompt cache and history intact. Traces in
``.governor/runs/`` stay the record of what happened; this is the record of the
conversation. ``resume_command`` is the ``/chats`` command, registered in ``features.py``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_core import Message, Usage

if TYPE_CHECKING:
    from .session import Session


def title_from(task: str, limit: int = 60) -> str:
    first = task.strip().splitlines()[0] if task.strip() else "(empty)"
    return first if len(first) <= limit else first[: limit - 1] + "…"


@dataclass
class Chat:
    id: str
    title: str
    provider: str
    model: str
    """The model alias the chat ran on (``openai/astra``), for the prompt cache and /qa."""
    agent: str
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    @classmethod
    def new(cls, task: str, provider: str, model: str, agent: str) -> Chat:
        return cls(
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4], title_from(task), provider, model, agent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "provider": self.provider,
            "model": self.model,
            "agent": self.agent,
            "created": self.created,
            "updated": self.updated,
            "usage": self.usage.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chat:
        return cls(
            id=data["id"],
            title=data["title"],
            provider=data["provider"],
            model=data["model"],
            agent=data.get("agent", "general"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            usage=Usage(**data.get("usage", {})),
            created=data.get("created", 0.0),
            updated=data.get("updated", 0.0),
        )

    @property
    def turns(self) -> int:
        """User prompts in the chat (tool-result messages don't count)."""
        return sum(1 for m in self.messages if m.role == "user" and m.text.strip() and not _is_tool_results(m))


def _is_tool_results(message: Message) -> bool:
    return bool(message.content) and all(b.type == "tool_result" for b in message.content)


class ChatStore:
    def __init__(self, project_dir: Path) -> None:
        self.dir = project_dir / ".governor" / "chats"

    def path(self, chat_id: str) -> Path:
        return self.dir / f"{chat_id}.json"

    def save(self, chat: Chat) -> Path:
        chat.updated = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path(chat.id)
        path.write_text(json.dumps(chat.to_dict(), ensure_ascii=False, default=str))
        return path

    def load(self, chat_id: str) -> Chat:
        return Chat.from_dict(json.loads(self.path(chat_id).read_text()))

    def list(self, limit: int = 30) -> list[Chat]:
        """Saved chats, newest first. Unreadable files are skipped."""
        chats = []
        for path in self.dir.glob("*.json") if self.dir.is_dir() else []:
            try:
                chats.append(Chat.from_dict(json.loads(path.read_text())))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        chats.sort(key=lambda c: c.updated, reverse=True)
        return chats[:limit]


def resume_command(session: Session, arg: str) -> None:
    """``/chats``: list the project's saved chats and make one the conversation again, transcript included."""
    chats = session.chats.list() if session.chats is not None else []
    if not chats:
        session.say("no saved chats in this project yet; each completed turn saves one", "dim")
        return
    chosen = next((c for c in chats if c.id == arg), None) if arg else None
    if chosen is None:
        options = [
            (
                c.id,
                f"{time.strftime('%b %d %H:%M', time.localtime(c.updated))}  {c.title:<40.40} {c.turns:>3} turns  {c.model}",
            )
            for c in chats
        ]
        picked = session.choose("Resume a chat", options, default=session.chat.id if session.chat else None)
        if picked is None:
            return
        chosen = next(c for c in chats if c.id == picked)
    resumed = session.resume(chosen)
    if session.io is not None:
        session.io.clear()
    session.say(f"resumed: {chosen.title}", "dim")
    for m in resumed.messages:
        if m.role == "user":
            if m.content and m.content[0].type == "text" and m.content[0].text.strip():
                session.say("› " + m.content[0].text, "user")
        elif m.text.strip():
            session.say(m.text, "answer")
    session.say(resumed.note, "dim")
