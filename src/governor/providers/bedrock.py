"""Amazon Bedrock via the Converse API (boto3 ``bedrock-runtime``).

Converse uses one request shape for every model family on Bedrock (Claude,
Nova, Llama, Qwen, DeepSeek...), so routing between models is just a
different model id.

Auth goes through the standard AWS credential chain: ``AWS_PROFILE``
(including SSO profiles), access keys, instance roles, or a Bedrock API key
in ``AWS_BEARER_TOKEN_BEDROCK``. The region comes from ``GOVERNOR_AWS_REGION``,
then ``AWS_REGION``, then the profile, with ``us-east-1`` as the fallback.

Prompt caching on Converse needs explicit cache points, and a request may have
four: after the system prompt, after the tools, and at the end of the last two
messages. The last one caches the conversation so far, so the next turn reads it
back instead of paying full price for it again. The one before it marks where
later requests may branch off (``/qa`` sends questions after the conversation
as it was), so each of those reads the same entry.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

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

NAME = "bedrock"
_CACHE_POINT = {"cachePoint": {"type": "default"}}

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "content_filtered": StopReason.CONTENT_FILTERED,
    "guardrail_intervened": StopReason.CONTENT_FILTERED,
}


def aws_session() -> boto3.Session:
    profile = os.environ.get("GOVERNOR_AWS_PROFILE") or os.environ.get("AWS_PROFILE")
    region = os.environ.get("GOVERNOR_AWS_REGION") or os.environ.get("AWS_REGION")
    session = boto3.Session(profile_name=profile, region_name=region)
    if session.region_name is None:
        session = boto3.Session(profile_name=profile, region_name="us-east-1")
    return session


def explain_error(exc: Exception) -> ProviderError:
    """Map common Bedrock failures to an actionable hint."""
    if isinstance(exc, NoCredentialsError):
        return ProviderError(
            "no AWS credentials found",
            hint="sign in with /login bedrock",
        )
    if not isinstance(exc, ClientError):
        text = str(exc)
        if "crt" in text:
            return ProviderError(
                f"AWS credentials: {text}", hint="pip install 'botocore[crt]' (needed for `aws login`)"
            )
        if any(s in text.lower() for s in ("expired", "reauthenticate", "token", "sso")):
            return ProviderError(
                f"AWS credentials: {text}",
                hint="your AWS sign-in expired; sign in again with /login bedrock",
            )
        return ProviderError(f"Bedrock call failed: {text}", retryable=True)
    err = exc.response.get("Error", {})
    code, msg = err.get("Code", ""), err.get("Message", "")
    lowered = msg.lower()
    hint = ""
    if "on-demand throughput" in lowered or "inference profile" in lowered:
        hint = "use a cross-region inference profile id (us./global. prefix); /model → Other lists them"
    elif code == "AccessDeniedException" and "not available for this account" in lowered:
        hint = (
            "the account has no usable agreement for this model (Bedrock reports agreementAvailability="
            "NOT_AVAILABLE; IAM is not the problem). Ask AWS Support/Sales to enable it, or use a model whose "
            "agreement is AVAILABLE (`/doctor` shows the status)"
        )
    elif code == "AccessDeniedException" and ("use case" in lowered or "model access" in lowered):
        hint = "enable model access / submit the Anthropic use-case form in the Bedrock console"
    elif code == "AccessDeniedException":
        hint = "IAM needs bedrock:InvokeModel* on inference-profile/* and foundation-model/* in all routed regions"
    elif code == "ResourceNotFoundException" or "model identifier is invalid" in lowered:
        hint = "model id not found in this region; /model → Other lists the ids you can use"
    elif code in ("ExpiredTokenException", "UnrecognizedClientException"):
        hint = "credentials expired or invalid; sign in again with /login bedrock"
    elif code == "ThrottlingException":
        hint = "rate limited; retries are automatic, so raise the Bedrock quota if this keeps happening"
    retryable = code in ("ThrottlingException", "ServiceUnavailableException", "InternalServerException")
    return ProviderError(f"Bedrock {code}: {msg}", hint=hint, retryable=retryable)


class BedrockProvider:
    name = NAME

    def __init__(self, client: Any = None) -> None:
        if client is None:
            client = aws_session().client(
                "bedrock-runtime",
                config=Config(
                    retries={"max_attempts": 8, "mode": "adaptive"},
                    read_timeout=900,
                    connect_timeout=10,
                ),
            )
        self.client = client

    def complete(
        self,
        request: ModelRequest,
        *,
        on_delta: DeltaCallback | None = None,
        stop: threading.Event | None = None,
    ) -> ModelResponse:
        kwargs = to_converse(request)
        started = time.monotonic()
        try:
            if on_delta is None and stop is None:
                resp = self.client.converse(**kwargs)
                return from_converse(resp, request.model, time.monotonic() - started)
            streamed = self.client.converse_stream(**kwargs)
            return collect_stream(streamed["stream"], request.model, started, on_delta, stop)
        except (ClientError, BotoCoreError) as exc:
            raise explain_error(exc) from exc


# --- translation -------------------------------------------------------------


def _block_to_converse(block: Block) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        # Converse rejects blank text blocks.
        return {"text": block.text} if block.text.strip() else None
    if isinstance(block, ToolUseBlock):
        return {"toolUse": {"toolUseId": block.id, "name": block.name, "input": block.input}}
    if isinstance(block, ToolResultBlock):
        return {
            "toolResult": {
                "toolUseId": block.tool_use_id,
                "content": [{"text": block.content or "(no output)"}],
                "status": "error" if block.is_error else "success",
            }
        }
    if isinstance(block, ReasoningBlock):
        return block.raw if block.provider == NAME else None
    raise TypeError(f"unsupported block {block!r}")


def to_converse(request: ModelRequest) -> dict[str, Any]:
    messages = []
    for msg in request.messages:
        content = [c for c in (_block_to_converse(b) for b in msg.content) if c is not None]
        if content:
            messages.append({"role": msg.role, "content": content})
    if request.cache:
        for message in messages[-2:]:
            message["content"].append(_CACHE_POINT)

    kwargs: dict[str, Any] = {
        "modelId": request.model,
        "messages": messages,
        "inferenceConfig": {"maxTokens": request.max_tokens},
    }
    if request.temperature is not None:
        kwargs["inferenceConfig"]["temperature"] = request.temperature
    if request.system:
        kwargs["system"] = [{"text": request.system}] + ([_CACHE_POINT] if request.cache else [])
    if request.tools:
        tools: list[dict[str, Any]] = [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": t.input_schema},
                }
            }
            for t in request.tools
        ]
        if request.cache:
            tools.append(_CACHE_POINT)
        kwargs["toolConfig"] = {"tools": tools}
    if request.params:
        kwargs["additionalModelRequestFields"] = request.params
    if request.output_schema is not None:
        # Converse's structured output: the schema travels as a JSON string.
        kwargs["outputConfig"] = {
            "textFormat": {
                "type": "json_schema",
                "structure": {"jsonSchema": {"schema": json.dumps(request.output_schema), "name": "output"}},
            }
        }
    return kwargs


def from_converse(resp: dict[str, Any], model: str, latency_s: float) -> ModelResponse:
    blocks: list[Block] = []
    for c in resp["output"]["message"]["content"]:
        if "text" in c:
            blocks.append(TextBlock(c["text"]))
        elif "toolUse" in c:
            tu = c["toolUse"]
            blocks.append(ToolUseBlock(tu["toolUseId"], tu["name"], tu.get("input") or {}))
        elif "reasoningContent" in c:
            blocks.append(ReasoningBlock(raw=c, provider=NAME))
    u = resp.get("usage", {})
    usage = Usage(
        input_tokens=u.get("inputTokens", 0),
        output_tokens=u.get("outputTokens", 0),
        cache_read_tokens=u.get("cacheReadInputTokens", 0),
        cache_write_tokens=u.get("cacheWriteInputTokens", 0),
    )
    return ModelResponse(
        message=Message("assistant", tuple(blocks)),
        stop_reason=_STOP_REASONS.get(resp.get("stopReason", ""), StopReason.OTHER),
        usage=usage,
        model=model,
        latency_s=latency_s,
        raw={"metrics": resp.get("metrics", {}), "stopReason": resp.get("stopReason")},
    )


# --- streaming ---------------------------------------------------------------


def _usage(u: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=u.get("inputTokens", 0),
        output_tokens=u.get("outputTokens", 0),
        cache_read_tokens=u.get("cacheReadInputTokens", 0),
        cache_write_tokens=u.get("cacheWriteInputTokens", 0),
    )


def _assemble(parts: dict[int, dict[str, Any]]) -> Message:
    blocks: list[Block] = []
    for _, part in sorted(parts.items()):
        kind = part["kind"]
        if kind == "text":
            blocks.append(TextBlock("".join(part["text"])))
        elif kind == "tool_use":
            raw = "".join(part["json"])
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {}  # the tool's schema check tells the model what is missing
            blocks.append(ToolUseBlock(part["id"], part["name"], parsed if isinstance(parsed, dict) else {}))
        elif kind == "reasoning":
            if part.get("redacted") is not None:
                content: dict[str, Any] = {"redactedContent": part["redacted"]}
            else:
                content = {"reasoningText": {"text": "".join(part["text"])}}
                if part.get("signature"):
                    content["reasoningText"]["signature"] = part["signature"]
            blocks.append(ReasoningBlock(raw={"reasoningContent": content}, provider=NAME))
    return Message("assistant", tuple(blocks))


def _close(events: Any) -> None:
    raw = getattr(events, "_raw_stream", None)
    if raw is not None and hasattr(raw, "close"):
        raw.close()


def collect_stream(
    events: Iterable[dict[str, Any]],
    model: str,
    started: float,
    on_delta: DeltaCallback | None,
    stop: threading.Event | None,
) -> ModelResponse:
    """Fold ConverseStream events into one response, reporting fragments as they arrive."""
    parts: dict[int, dict[str, Any]] = {}
    usage, stop_reason, metrics = Usage(), StopReason.OTHER, {}
    for event in events:
        if stop is not None and stop.is_set():
            _close(events)
            raise ProviderCancelled(partial=_assemble(parts))
        if "contentBlockStart" in event:
            start, i = event["contentBlockStart"]["start"], event["contentBlockStart"]["contentBlockIndex"]
            if "toolUse" in start:
                parts[i] = {
                    "kind": "tool_use",
                    "id": start["toolUse"]["toolUseId"],
                    "name": start["toolUse"]["name"],
                    "json": [],
                }
        elif "contentBlockDelta" in event:
            delta, i = event["contentBlockDelta"]["delta"], event["contentBlockDelta"]["contentBlockIndex"]
            if "text" in delta:
                parts.setdefault(i, {"kind": "text", "text": []})["text"].append(delta["text"])
                if on_delta:
                    on_delta(Delta("text", delta["text"], i))
            elif "toolUse" in delta:
                parts.setdefault(i, {"kind": "tool_use", "id": "", "name": "", "json": []})["json"].append(
                    delta["toolUse"].get("input", "")
                )
                if on_delta:
                    on_delta(Delta("tool_input", delta["toolUse"].get("input", ""), i))
            elif "reasoningContent" in delta:
                r = delta["reasoningContent"]
                part = parts.setdefault(i, {"kind": "reasoning", "text": [], "signature": None, "redacted": None})
                if "text" in r:
                    part["text"].append(r["text"])
                    if on_delta:
                        on_delta(Delta("reasoning", r["text"], i))
                if "signature" in r:
                    part["signature"] = r["signature"]
                if "redactedContent" in r:
                    part["redacted"] = r["redactedContent"]
        elif "messageStop" in event:
            stop_reason = _STOP_REASONS.get(event["messageStop"].get("stopReason", ""), StopReason.OTHER)
        elif "metadata" in event:
            usage = _usage(event["metadata"].get("usage", {}))
            metrics = event["metadata"].get("metrics", {})
    return ModelResponse(
        message=_assemble(parts),
        stop_reason=stop_reason,
        usage=usage,
        model=model,
        latency_s=time.monotonic() - started,
        raw={"metrics": metrics, "stopReason": stop_reason.value},
    )
