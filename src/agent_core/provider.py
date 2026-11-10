"""The provider interface: one synchronous call, neutral types in and out.

``complete`` may stream: when the loop passes ``on_delta``, a provider that
can stream calls it with each fragment (text, reasoning or tool-input JSON)
as it arrives and still returns the whole response at the end. When ``stop``
is given and gets set during the call, the provider closes the connection and
raises ``ProviderCancelled`` with whatever had arrived. Both are optional:
a provider that ignores them is still correct, just not live.
"""

from __future__ import annotations

import threading
from typing import Protocol

from .types import DeltaCallback, Message, ModelRequest, ModelResponse


class ProviderError(RuntimeError):
    """A model call failed. ``hint`` says what to do about it, when known."""

    def __init__(self, message: str, *, hint: str = "", retryable: bool = False) -> None:
        super().__init__(f"{message}\n  hint: {hint}" if hint else message)
        self.hint = hint
        self.retryable = retryable


class ProviderCancelled(ProviderError):
    """The call was stopped on request. ``partial`` is what the model had produced so far, if anything."""

    def __init__(self, partial: Message | None = None) -> None:
        super().__init__("stopped by user")
        self.partial = partial


class Provider(Protocol):
    name: str

    def complete(
        self,
        request: ModelRequest,
        *,
        on_delta: DeltaCallback | None = None,
        stop: threading.Event | None = None,
    ) -> ModelResponse:
        """Run one model call. Raises ProviderError on failure, ProviderCancelled when ``stop`` is set."""
        ...
