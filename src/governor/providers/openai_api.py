"""OpenAI models through OpenAI's own API, via the Responses API.

Responses is OpenAI's API for reasoning models with tools. The provider runs it
statelessly (``store=False``): the full history is sent every turn, as with the
other providers, and reasoning comes back encrypted
(``include=["reasoning.encrypted_content"]``) so it can be passed back verbatim
as a ``ReasoningBlock``, keeping history append-only.

Auth is ``OPENAI_API_KEY`` (``/login openai`` saves it);
``OPENAI_BASE_URL`` points it at a proxy or another Responses-compatible server.
``params`` from an agent profile are merged into the request last, so
``reasoning: {effort: high}`` or ``include: []`` (for models without reasoning)
can be set per agent.

Prompt caching is automatic on OpenAI's side, so ``ModelRequest.cache`` has no
effect here. OpenAI counts cached tokens inside ``input_tokens``; they are moved
out into ``cache_read_tokens``/``cache_write_tokens`` to match the other providers.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from agent_core import (
    Block,
    Delta,
    DeltaCallback,
    Message,
    ModelRequest,
    ModelResponse,
    ProviderCancelled,
    ProviderError,
    ReasoningBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

NAME = "openai"

_INCOMPLETE_REASONS = {
    "max_output_tokens": StopReason.MAX_TOKENS,
    "content_filter": StopReason.CONTENT_FILTERED,
}


def _hint(status: int) -> str:
    if status == 401:
        return "the key is wrong or revoked; sign in again with /login openai"
    if status == 403:
        return "this key's project or organization can't use this model (some need organization verification)"
    if status == 404:
        return "unknown model id, or no access to it; OpenAI ids have no prefix (gpt-6-astra); pick one with /model"
    if status == 429:
        return "rate limited or out of quota; the SDK retries automatically, so check the org's limits and billing"
    if status == 400:
        return "check the agent's `params`; models without reasoning may need `include: []`"
    return ""


class OpenAIProvider:
    name = NAME

    def __init__(self, client: Any = None) -> None:
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise ProviderError(
                    "the openai provider needs the openai SDK",
                    hint="reinstall Governor: pip install -e .",
                ) from exc
            try:
                client = openai.OpenAI(max_retries=5)
            except openai.OpenAIError as exc:
                raise ProviderError(f"OpenAI client error: {exc}", hint="sign in with /login openai") from exc
        self.client = client

    def complete(
        self,
        request: ModelRequest,
        *,
        on_delta: DeltaCallback | None = None,
        stop: threading.Event | None = None,
    ) -> ModelResponse:
        import openai

        kwargs = to_responses_api(request)
        started = time.monotonic()
        try:
            if on_delta is None and stop is None:
                resp = self.client.responses.create(**kwargs)
            else:
                resp = _stream(self.client.responses.create(stream=True, **kwargs), on_delta, stop)
        except openai.APIStatusError as exc:
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise ProviderError(
                f"OpenAI {exc.status_code}: {exc.message}", hint=_hint(exc.status_code), retryable=retryable
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"OpenAI connection error: {exc}", retryable=True) from exc
        if resp.status == "failed":
            error = getattr(resp, "error", None)
            raise ProviderError(f"OpenAI response failed: {getattr(error, 'message', error)}", retryable=True)
        return from_responses_api(resp, request.model, time.monotonic() - started)


# --- translation -------------------------------------------------------------


def _items(msg: Message) -> list[dict[str, Any]]:
    """One core message becomes several Responses input items, in order."""
    items: list[dict[str, Any]] = []
    text: list[str] = []

    def flush() -> None:
        if text:
            joined = "\n\n".join(text)
            if msg.role == "user":
                items.append({"role": "user", "content": joined})
            else:
                items.append(
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": joined}]}
                )
            text.clear()

    for block in msg.content:
        if isinstance(block, TextBlock):
            if block.text.strip():
                text.append(block.text)
            continue
        flush()
        if isinstance(block, ToolUseBlock):
            items.append(
                {"type": "function_call", "call_id": block.id, "name": block.name, "arguments": json.dumps(block.input)}
            )
        elif isinstance(block, ToolResultBlock):
            output = block.content or "(no output)"
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    # Responses has no error flag on tool output, so say it in the text.
                    "output": f"[tool error] {output}" if block.is_error else output,
                }
            )
        elif isinstance(block, ReasoningBlock):
            if block.provider == NAME:
                items.append(block.raw)
        else:
            raise TypeError(f"unsupported block {block!r}")
    flush()
    return items


def to_responses_api(request: ModelRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "input": [item for msg in request.messages for item in _items(msg)],
        "max_output_tokens": request.max_tokens,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if request.system:
        kwargs["instructions"] = request.system
    if request.tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
                "strict": False,
            }
            for t in request.tools
        ]
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    kwargs.update(request.params)
    if request.output_schema is not None:
        text = dict(kwargs.get("text") or {})  # params may already set verbosity there
        text["format"] = {"type": "json_schema", "name": "output", "schema": request.output_schema, "strict": False}
        kwargs["text"] = text
    return kwargs


def _stream(events: Any, on_delta: DeltaCallback | None, stop: threading.Event | None) -> Any:
    """Consume a Responses stream; returns the final response object from its last event."""
    final, parts = None, []
    for event in events:
        if stop is not None and stop.is_set():
            events.close()
            partial = Message("assistant", (TextBlock("".join(parts)),)) if parts else None
            raise ProviderCancelled(partial=partial)
        kind = getattr(event, "type", "")
        if kind == "response.output_text.delta":
            parts.append(event.delta)
            if on_delta:
                on_delta(Delta("text", event.delta, getattr(event, "output_index", 0)))
        elif kind == "response.function_call_arguments.delta":
            if on_delta:
                on_delta(Delta("tool_input", event.delta, getattr(event, "output_index", 0)))
        elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            if on_delta:
                on_delta(Delta("reasoning", event.delta, getattr(event, "output_index", 0)))
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            final = event.response
    if final is None:
        raise ProviderError("OpenAI stream ended without a final response", retryable=True)
    return final


def _arguments(raw: str) -> dict[str, Any]:
    # Truncated or malformed arguments become {}; the tool then reports the missing fields.
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def from_responses_api(resp: Any, model: str, latency_s: float) -> ModelResponse:
    blocks: list[Block] = []
    refused = False
    for item in resp.output:
        if item.type == "reasoning":
            blocks.append(ReasoningBlock(raw=item.model_dump(exclude_none=True, by_alias=True), provider=NAME))
        elif item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    blocks.append(TextBlock(part.text))
                elif part.type == "refusal":
                    blocks.append(TextBlock(part.refusal))
                    refused = True
        elif item.type == "function_call":
            blocks.append(ToolUseBlock(item.call_id, item.name, _arguments(item.arguments)))

    incomplete = getattr(resp.incomplete_details, "reason", None) if resp.incomplete_details else None
    if resp.status == "incomplete":
        stop = _INCOMPLETE_REASONS.get(incomplete or "", StopReason.OTHER)
    elif any(isinstance(b, ToolUseBlock) for b in blocks):
        stop = StopReason.TOOL_USE
    elif refused:
        stop = StopReason.CONTENT_FILTERED
    else:
        stop = StopReason.END_TURN

    u = resp.usage
    details = getattr(u, "input_tokens_details", None)
    cache_read = getattr(details, "cached_tokens", 0) or 0
    cache_write = getattr(details, "cache_write_tokens", 0) or 0
    usage = Usage(
        input_tokens=max(0, (getattr(u, "input_tokens", 0) or 0) - cache_read - cache_write),
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
    return ModelResponse(
        message=Message("assistant", tuple(blocks)),
        stop_reason=stop,
        usage=usage,
        model=model,
        latency_s=latency_s,
        raw={"id": resp.id, "status": resp.status, "incomplete_reason": incomplete},
    )
