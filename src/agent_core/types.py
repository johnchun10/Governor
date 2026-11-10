"""Provider-neutral message and model types.

The shapes mirror Bedrock's Converse API closely (content blocks, tool_use /
tool_result pairs) because Converse is itself a neutral format across model
families. Providers translate to and from these; nothing above the provider
layer should ever see an SDK object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTERED = "content_filtered"
    OTHER = "other"


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True)
class ReasoningBlock:
    """Opaque model reasoning (thinking) block, carried verbatim.

    Recent Claude models require thinking blocks to be sent back unchanged on
    later turns, so Governor never edits or drops them: history is
    append-only. ``provider`` records which provider produced ``raw`` so a
    different provider can skip it rather than send a foreign shape.
    """

    raw: dict[str, Any]
    provider: str
    type: Literal["reasoning"] = "reasoning"


Block = TextBlock | ToolUseBlock | ToolResultBlock | ReasoningBlock


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: tuple[Block, ...]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", (TextBlock(text),))

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": [block_to_dict(b) for b in self.content]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(data["role"], tuple(block_from_dict(b) for b in data["content"]))


def block_to_dict(block: Block) -> dict[str, Any]:
    return dict(block.__dict__)


def block_from_dict(data: dict[str, Any]) -> Block:
    kind = data["type"]
    fields = {k: v for k, v in data.items() if k != "type"}
    if kind == "text":
        return TextBlock(**fields)
    if kind == "tool_use":
        return ToolUseBlock(**fields)
    if kind == "tool_result":
        return ToolResultBlock(**fields)
    if kind == "reasoning":
        return ReasoningBlock(**fields)
    raise ValueError(f"unknown block type: {kind}")


@dataclass(frozen=True)
class ToolSpec:
    """What the model sees about a tool: name, description, JSON schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: tuple[Message, ...]
    system: str = ""
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int = 8192
    temperature: float | None = None
    cache: bool = True
    """Ask the provider to insert prompt-cache points (system + tools) when supported."""
    params: dict[str, Any] = field(default_factory=dict)
    """Provider-specific extras passed through untouched (e.g. Converse
    ``additionalModelRequestFields``, or Messages API ``output_config``)."""
    output_schema: dict[str, Any] | None = None
    """Ask for the reply as JSON matching this JSON Schema (the provider's native
    structured-output mode). The caller still validates the text it gets back:
    see ``agent_core.schema`` and ``AgentSpec.output_schema``."""


@dataclass(frozen=True)
class Delta:
    """One streamed fragment of a model reply, delivered while the call runs."""

    kind: Literal["text", "reasoning", "tool_input"]
    text: str
    index: int = 0
    """Which content block of the reply the fragment belongs to."""


DeltaCallback = Callable[[Delta], None]


@dataclass(frozen=True)
class ModelResponse:
    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
