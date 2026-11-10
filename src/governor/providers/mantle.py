"""Claude on Bedrock via the Anthropic Messages API ("Mantle" endpoint).

This is Anthropic's recommended Bedrock path for Claude. New Claude features
(effort, thinking display, betas) show up here in the native Messages shape.
Converse remains the default because it covers every model family.

It needs ``bedrock-mantle:*`` IAM
permissions, and model ids have no region prefix (``anthropic.claude-opus-5-5``).
Credentials come from the same AWS chain as the Converse provider.

Calls always stream (the SDK collects the final message), so large
``max_tokens`` values don't hit HTTP timeouts; fragments are reported through
``on_delta`` and a set ``stop`` event closes the stream. The translation here
is shared with the first-party ``anthropic`` provider.
"""

from __future__ import annotations

import os
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

NAME = "mantle"

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.CONTENT_FILTERED,
}


class MantleProvider:
    name = NAME

    def __init__(self, client: Any = None) -> None:
        if client is None:
            try:
                from anthropic import AnthropicBedrockMantle
            except ImportError as exc:
                raise ProviderError(
                    "the mantle provider needs the anthropic SDK",
                    hint="reinstall Governor: pip install -e .",
                ) from exc
            region = os.environ.get("GOVERNOR_AWS_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
            client = AnthropicBedrockMantle(aws_region=region)
        self.client = client

    def complete(
        self,
        request: ModelRequest,
        *,
        on_delta: DeltaCallback | None = None,
        stop: threading.Event | None = None,
    ) -> ModelResponse:
        import anthropic

        kwargs = to_messages_api(request)
        try:
            return run_stream(self.client.messages, kwargs, request.model, NAME, on_delta, stop)
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise ProviderError(f"Mantle {exc.status_code}: {exc.message}", retryable=retryable) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Mantle connection error: {exc}", retryable=True) from exc


# --- translation (shared with the first-party ``anthropic`` provider) ---------


def _block(block: Block, provider: str) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text} if block.text.strip() else None
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content or "(no output)",
            "is_error": block.is_error,
        }
    if isinstance(block, ReasoningBlock):
        # Thinking blocks go back verbatim; foreign-provider reasoning is skipped.
        return block.raw if block.provider == provider else None
    raise TypeError(f"unsupported block {block!r}")


def to_messages_api(request: ModelRequest, provider: str = NAME) -> dict[str, Any]:
    messages = []
    for msg in request.messages:
        content = [c for c in (_block(b, provider) for b in msg.content) if c is not None]
        if content:
            messages.append({"role": msg.role, "content": content})
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
        **request.params,
    }
    if request.system:
        kwargs["system"] = request.system
    if request.tools:
        kwargs["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in request.tools
        ]
    if request.cache:
        # Top-level cache_control caches through the last block. Also mark the end of the message before it,
        # where later requests may branch off (/qa questions follow the conversation as it was), so they all
        # read one entry instead of each writing its own.
        kwargs["cache_control"] = {"type": "ephemeral"}
        if len(messages) > 1:
            _mark_cache_point(messages[-2]["content"])
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.output_schema is not None:
        # Structured output lives beside `effort` in output_config, which a profile's params may already set.
        output_config = dict(kwargs.get("output_config") or {})
        output_config["format"] = {"type": "json_schema", "schema": request.output_schema}
        kwargs["output_config"] = output_config
    return kwargs


def _mark_cache_point(content: list[dict[str, Any]]) -> None:
    for i in range(len(content) - 1, -1, -1):
        if content[i].get("type") in ("text", "tool_use", "tool_result"):  # thinking blocks can't carry cache_control
            content[i] = {**content[i], "cache_control": {"type": "ephemeral"}}
            return


def from_messages_api(resp: Any, model: str, latency_s: float, provider: str = NAME) -> ModelResponse:
    blocks: list[Block] = []
    for b in resp.content:
        if b.type == "text":
            blocks.append(TextBlock(b.text))
        elif b.type == "tool_use":
            blocks.append(ToolUseBlock(b.id, b.name, dict(b.input or {})))
        elif b.type in ("thinking", "redacted_thinking"):
            blocks.append(ReasoningBlock(raw=b.to_dict(), provider=provider))
    u = resp.usage
    usage = Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )
    return ModelResponse(
        message=Message("assistant", tuple(blocks)),
        stop_reason=_STOP_REASONS.get(resp.stop_reason or "", StopReason.OTHER),
        usage=usage,
        model=model,
        latency_s=latency_s,
        raw={"id": resp.id, "stop_reason": resp.stop_reason},
    )


def _partial(stream: Any, model: str, provider: str) -> Message | None:
    try:
        snapshot = stream.current_message_snapshot
    except Exception:  # nothing arrived yet
        return None
    return from_messages_api(snapshot, model, 0.0, provider=provider).message


def run_stream(
    api: Any,
    kwargs: dict[str, Any],
    model: str,
    provider: str,
    on_delta: DeltaCallback | None,
    stop: threading.Event | None,
) -> ModelResponse:
    """Stream one Messages API call (``api`` is ``client.messages`` or ``client.beta.messages``)."""
    started = time.monotonic()
    with api.stream(**kwargs) as stream:
        for event in stream:
            if stop is not None and stop.is_set():
                stream.close()
                raise ProviderCancelled(partial=_partial(stream, model, provider))
            if on_delta is None or getattr(event, "type", "") != "content_block_delta":
                continue
            delta, index = event.delta, event.index
            if delta.type == "text_delta":
                on_delta(Delta("text", delta.text, index))
            elif delta.type == "thinking_delta":
                on_delta(Delta("reasoning", delta.thinking, index))
            elif delta.type == "input_json_delta":
                on_delta(Delta("tool_input", delta.partial_json, index))
        resp = stream.get_final_message()
    return from_messages_api(resp, model, time.monotonic() - started, provider=provider)
