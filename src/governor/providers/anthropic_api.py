"""Claude through Anthropic's own API (``api.anthropic.com``), billed to an Anthropic account.

Same Messages API shape as the Mantle provider, so it reuses that translation;
only the client, auth and error hints differ. Useful when a Claude feature
reaches the first-party API before Bedrock, or to run without AWS at all.

Credentials resolve the way the SDK does: ``ANTHROPIC_API_KEY``, then ``ANTHROPIC_AUTH_TOKEN``, then an
``ant auth login`` profile. ``ANTHROPIC_BASE_URL`` points it at a proxy.

Requests stream (and are collected into one response) so large ``max_tokens``
values don't hit HTTP timeouts; see ``mantle.run_stream`` for the fragments and
the stop event. A ``betas`` list in ``params`` (e.g. from an agent profile)
routes the call through ``client.beta.messages``.
"""

from __future__ import annotations

import threading
from typing import Any

from agent_core import DeltaCallback, ModelRequest, ModelResponse, ProviderError

from .mantle import run_stream, to_messages_api

NAME = "anthropic"


def _hint(status: int, message: str) -> str:
    lowered = message.lower()
    if status == 401:
        return "the key is wrong or revoked; sign in again with /login anthropic"
    if status == 403:
        return "this key's workspace or organization can't use this model or feature"
    if status == 404:
        return "unknown model id; first-party ids have no prefix (claude-opus-5-5); pick one with /model"
    if status == 429:
        return "rate limited; the SDK retries automatically, so raise the org's rate limits if this keeps happening"
    if status == 400 and "credit balance" in lowered:
        return "the Anthropic account is out of credits; add some under Billing in the Claude Console"
    if status == 400 and ("thinking" in lowered or "tool_choice" in lowered):
        return "check the agent's `params` against this model's API (some thinking/tool_choice modes 400 on new models)"
    return ""


class AnthropicProvider:
    name = NAME

    def __init__(self, client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError(
                    "the anthropic provider needs the anthropic SDK",
                    hint="reinstall Governor: pip install -e .",
                ) from exc
            client = anthropic.Anthropic(max_retries=5)
        self.client = client

    def complete(
        self,
        request: ModelRequest,
        *,
        on_delta: DeltaCallback | None = None,
        stop: threading.Event | None = None,
    ) -> ModelResponse:
        import anthropic

        kwargs = to_messages_api(request, provider=NAME)
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        try:
            return run_stream(api, kwargs, request.model, NAME, on_delta, stop)
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise ProviderError(
                f"Anthropic {exc.status_code}: {exc.message}",
                hint=_hint(exc.status_code, exc.message),
                retryable=retryable,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Anthropic connection error: {exc}", retryable=True) from exc
        except TypeError as exc:
            # The SDK raises TypeError, before sending anything, when it finds no credentials.
            if "authentication" not in str(exc):
                raise
            raise ProviderError("no Anthropic credentials found", hint="sign in with /login anthropic") from exc
