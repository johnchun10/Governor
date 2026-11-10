"""ScriptedProvider: replays queued responses. For tests and offline development."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterable
from typing import Any

from .provider import ProviderCancelled
from .types import (
    Block,
    Delta,
    DeltaCallback,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)

_ids = itertools.count(1)


def text(s: str, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(Message("assistant", (TextBlock(s),)), StopReason.END_TURN, usage or Usage(10, 5), "fake")


def tool_call(
    name: str, args: dict[str, Any] | None = None, *, say: str = "", usage: Usage | None = None
) -> ModelResponse:
    return tool_calls([(name, args or {})], say=say, usage=usage)


def tool_calls(
    calls: Iterable[tuple[str, dict[str, Any]]], *, say: str = "", usage: Usage | None = None
) -> ModelResponse:
    blocks: list[Block] = [TextBlock(say)] if say else []
    blocks += [ToolUseBlock(f"tu_{next(_ids)}", name, args) for name, args in calls]
    return ModelResponse(Message("assistant", tuple(blocks)), StopReason.TOOL_USE, usage or Usage(10, 5), "fake")


Script = ModelResponse | Callable[[ModelRequest], ModelResponse]


class ScriptedProvider:
    """Returns scripted responses in order.

    Pass a list for a single linear conversation, or a ``router`` callable
    that picks a response from the request (useful when parallel subagents
    call the same provider in nondeterministic order).
    """

    name = "fake"

    def __init__(
        self, script: Iterable[Script] = (), router: Callable[[ModelRequest], ModelResponse] | None = None
    ) -> None:
        self._script = list(script)
        self._router = router
        self._lock = threading.Lock()
        self.requests: list[ModelRequest] = []

    def complete(
        self,
        request: ModelRequest,
        *,
        on_delta: DeltaCallback | None = None,
        stop: threading.Event | None = None,
    ) -> ModelResponse:
        with self._lock:
            self.requests.append(request)
        if self._router is not None:
            response = self._router(request)  # outside the lock so parallel callers overlap
        else:
            with self._lock:
                if not self._script:
                    raise AssertionError("ScriptedProvider ran out of responses")
                item = self._script.pop(0)
            response = item(request) if callable(item) else item
        if stop is not None and stop.is_set():
            raise ProviderCancelled()
        if on_delta is not None:
            # "Stream" the scripted reply: its text in two pieces, so callers see more than one delta.
            for i, block in enumerate(response.message.content):
                if isinstance(block, TextBlock) and block.text:
                    half = max(1, len(block.text) // 2)
                    on_delta(Delta("text", block.text[:half], i))
                    on_delta(Delta("text", block.text[half:], i))
        return response
