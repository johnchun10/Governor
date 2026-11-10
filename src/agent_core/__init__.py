"""agent_core: the agent primitives. Stable contracts plus the reference loop.

Everything here is standard library only and knows nothing about Bedrock, the
CLI, specific tools or profiles. Those live in ``governor``, which depends on
this package and never the other way round. See ``agent_core/README.md`` for
what belongs here and how changes to it are reviewed.
"""

from .agent import Agent, AgentSpec, RunResult, Runtime
from .context import ContextProvider, RunView
from .events import SCHEMA_VERSION, Collector, Event, EventBus, Sink
from .models import ModelInfo
from .permissions import Decision, PermissionPolicy, StaticPolicy
from .provider import Provider, ProviderCancelled, ProviderError
from .schema import parse_json, validate
from .subagent import task_tool
from .tools import Tool, ToolContext, ToolError, ToolRegistry, schema_from_function, tool
from .types import (
    Block,
    Delta,
    DeltaCallback,
    Message,
    ModelRequest,
    ModelResponse,
    ReasoningBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

__version__ = "0.2.0"
"""Core API version. Bump it in every core-change PR (see README.md)."""

__all__ = [
    "SCHEMA_VERSION",
    "Agent",
    "AgentSpec",
    "Block",
    "Collector",
    "ContextProvider",
    "Decision",
    "Delta",
    "DeltaCallback",
    "Event",
    "EventBus",
    "Message",
    "ModelInfo",
    "ModelRequest",
    "ModelResponse",
    "PermissionPolicy",
    "Provider",
    "ProviderCancelled",
    "ProviderError",
    "ReasoningBlock",
    "RunResult",
    "RunView",
    "Runtime",
    "Sink",
    "StaticPolicy",
    "StopReason",
    "TextBlock",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "Usage",
    "parse_json",
    "schema_from_function",
    "task_tool",
    "tool",
    "validate",
]
