"""First-party API providers (Anthropic, OpenAI): request/response mapping with fake clients, no network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_core import (
    Message,
    ModelRequest,
    ProviderCancelled,
    ProviderError,
    ReasoningBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

from governor.providers.anthropic_api import AnthropicProvider  # noqa: E402
from governor.providers.mantle import to_messages_api  # noqa: E402
from governor.providers.openai_api import OpenAIProvider, to_responses_api  # noqa: E402

SPEC = ToolSpec(
    "read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
)


def _history(own: ReasoningBlock, foreign: ReasoningBlock) -> tuple[Message, ...]:
    return (
        Message.user("read a.txt"),
        Message("assistant", (own, foreign, TextBlock("Reading."), ToolUseBlock("t1", "read_file", {"path": "a.txt"}))),
        Message("user", (ToolResultBlock("t1", "boom", is_error=True), TextBlock("try again"))),
    )


# --- Anthropic -----------------------------------------------------------------


class _Stream:
    """A fake MessageStream: yields the given events, then hands back the final message."""

    def __init__(self, message, events=(), on_event=None):
        self.message, self.events, self.on_event, self.closed = message, list(events), on_event, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for event in self.events:
            if self.on_event:
                self.on_event(event)
            yield event

    def close(self):
        self.closed = True

    @property
    def current_message_snapshot(self):
        return self.message

    def get_final_message(self):
        return self.message


class _Messages:
    def __init__(self, message=None, error=None, events=(), on_event=None):
        self.message, self.error, self.calls = message, error, []
        self.events, self.on_event, self.streams = events, on_event, []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        stream = _Stream(self.message, self.events, self.on_event)
        self.streams.append(stream)
        return stream


def _delta(kind, index=0, **fields):
    return SimpleNamespace(type="content_block_delta", index=index, delta=SimpleNamespace(type=kind, **fields))


def _anthropic_message():
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "t9", "name": "read_file", "input": {"path": "a"}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 20,
            },
        }
    )


def test_anthropic_request_and_response():
    messages, beta = _Messages(_anthropic_message()), _Messages(_anthropic_message())
    client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=beta))
    provider = AnthropicProvider(client=client)
    own = ReasoningBlock(raw={"type": "thinking", "thinking": "", "signature": "s"}, provider="anthropic")
    foreign = ReasoningBlock(raw={"type": "thinking", "thinking": "", "signature": "m"}, provider="mantle")
    req = ModelRequest(
        model="claude-opus-5-5",
        system="be good",
        tools=(SPEC,),
        messages=_history(own, foreign),
        params={"output_config": {"effort": "high"}},
    )

    resp = provider.complete(req)

    sent = messages.calls[0]
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert sent["output_config"] == {"effort": "high"}
    assistant = sent["messages"][1]["content"]
    assert assistant[0] == own.raw  # own thinking passed back verbatim
    assert len(assistant) == 3  # mantle's thinking skipped
    assert sent["messages"][2]["content"][0]["is_error"] is True
    assert assistant[-1]["cache_control"] == {"type": "ephemeral"}  # the message before the last one is marked
    assert all("cache_control" not in block for block in sent["messages"][2]["content"])
    assert resp.stop_reason is StopReason.TOOL_USE
    assert resp.message.tool_uses == [ToolUseBlock("t9", "read_file", {"path": "a"})]
    assert isinstance(resp.message.content[0], ReasoningBlock)
    assert resp.message.content[0].provider == "anthropic"
    assert (resp.usage.cache_read_tokens, resp.usage.cache_write_tokens) == (100, 20)
    assert not beta.calls

    provider.complete(ModelRequest(model="m", messages=(Message.user("hi"),), params={"betas": ["x-2026-01-01"]}))
    assert beta.calls[0]["betas"] == ["x-2026-01-01"]


def test_messages_api_cache_point_skips_thinking_and_copies_blocks():
    thinking = ReasoningBlock(raw={"type": "thinking", "thinking": "", "signature": "s"}, provider="mantle")
    history = (Message.user("q1"), Message("assistant", (TextBlock("a1"), thinking)), Message.user("q2"))
    sent = to_messages_api(ModelRequest(model="m", messages=history))
    assert sent["messages"][1]["content"] == [
        {"type": "text", "text": "a1", "cache_control": {"type": "ephemeral"}},
        thinking.raw,
    ]
    assert "cache_control" not in thinking.raw
    assert "cache_control" not in to_messages_api(ModelRequest(model="m", messages=history[:1]))["messages"][0]
    assert "cache_control" not in to_messages_api(ModelRequest(model="m", messages=history, cache=False))


def test_anthropic_streams_deltas_and_structured_output_config():
    events = [
        SimpleNamespace(type="message_start"),
        _delta("thinking_delta", 0, thinking="hmm"),
        _delta("text_delta", 1, text="Read"),
        _delta("text_delta", 1, text="ing."),
        _delta("input_json_delta", 2, partial_json='{"path"'),
        SimpleNamespace(type="message_stop"),
    ]
    messages = _Messages(_anthropic_message(), events=events)
    provider = AnthropicProvider(client=SimpleNamespace(messages=messages, beta=None))
    deltas = []
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    req = ModelRequest(
        model="m", messages=(Message.user("hi"),), output_schema=schema, params={"output_config": {"effort": "low"}}
    )
    resp = provider.complete(req, on_delta=deltas.append)
    assert [(d.kind, d.text, d.index) for d in deltas] == [
        ("reasoning", "hmm", 0),
        ("text", "Read", 1),
        ("text", "ing.", 1),
        ("tool_input", '{"path"', 2),
    ]
    assert resp.message.text == "Reading."
    assert messages.calls[0]["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": schema},
    }


def test_anthropic_stop_closes_the_stream_and_keeps_the_partial():
    import threading

    stop = threading.Event()
    events = [_delta("text_delta", 0, text="Read"), _delta("text_delta", 0, text="ing.")]
    messages = _Messages(_anthropic_message(), events=events, on_event=lambda e: stop.set())
    provider = AnthropicProvider(client=SimpleNamespace(messages=messages, beta=None))
    with pytest.raises(ProviderCancelled) as info:
        provider.complete(ModelRequest(model="m", messages=(Message.user("hi"),)), stop=stop)
    assert messages.streams[0].closed
    assert info.value.partial is not None and info.value.partial.text == "Reading."


def test_anthropic_missing_credentials_has_hint():
    error = TypeError("Could not resolve authentication method. Expected one of api_key...")
    client = SimpleNamespace(messages=_Messages(error=error))
    with pytest.raises(ProviderError, match="login anthropic"):
        AnthropicProvider(client=client).complete(ModelRequest(model="m", messages=(Message.user("hi"),)))


# --- OpenAI --------------------------------------------------------------------


REASONING_RAW = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "enc"}


def test_openai_request_mapping():
    own = ReasoningBlock(raw=REASONING_RAW, provider="openai")
    foreign = ReasoningBlock(raw={"type": "thinking"}, provider="anthropic")
    req = ModelRequest(
        model="gpt-6-astra",
        system="be good",
        tools=(SPEC,),
        messages=_history(own, foreign),
        params={"reasoning": {"effort": "high"}},
    )
    k = to_responses_api(req)
    assert k["instructions"] == "be good"
    assert k["store"] is False
    assert k["include"] == ["reasoning.encrypted_content"]
    assert k["reasoning"] == {"effort": "high"}
    assert k["tools"][0] == {
        "type": "function",
        "name": "read_file",
        "description": "Read a file.",
        "parameters": SPEC.input_schema,
        "strict": False,
    }
    assert [i.get("type", i.get("role")) for i in k["input"]] == [
        "user",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
        "user",
    ]
    assert k["input"][1] == REASONING_RAW  # own reasoning passed back verbatim; foreign skipped
    assert json.loads(k["input"][3]["arguments"]) == {"path": "a.txt"}
    assert k["input"][4] == {"type": "function_call_output", "call_id": "t1", "output": "[tool error] boom"}
    assert k["input"][5] == {"role": "user", "content": "try again"}


def _openai_response(output, status="completed", incomplete=None):
    from openai.types.responses.response import IncompleteDetails
    from openai.types.responses.response_output_item import ResponseOutputItem
    from pydantic import TypeAdapter

    adapter = TypeAdapter(ResponseOutputItem)
    return SimpleNamespace(
        id="resp_1",
        status=status,
        incomplete_details=IncompleteDetails(reason=incomplete) if incomplete else None,
        output=[adapter.validate_python(o) for o in output],
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(cached_tokens=100, cache_write_tokens=0),
        ),
    )


class _Responses:
    def __init__(self, response, events=None):
        self.response, self.events, self.calls = response, events, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _EventStream(self.events if self.events is not None else [], self.response)
        return self.response


class _EventStream:
    def __init__(self, events, response):
        self.events, self.response, self.closed = events, response, False

    def __iter__(self):
        yield from self.events
        yield SimpleNamespace(type="response.completed", response=self.response)

    def close(self):
        self.closed = True


def test_openai_response_mapping():
    response = _openai_response(
        [
            REASONING_RAW,
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Reading.", "annotations": []}],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path":"a"}',
            },
        ]
    )
    provider = OpenAIProvider(client=SimpleNamespace(responses=_Responses(response)))
    resp = provider.complete(ModelRequest(model="gpt-6-astra", messages=(Message.user("hi"),)))

    reasoning, text, call = resp.message.content
    assert isinstance(reasoning, ReasoningBlock)
    assert reasoning.provider == "openai"
    assert reasoning.raw["encrypted_content"] == "enc"
    assert text == TextBlock("Reading.")
    assert call == ToolUseBlock("call_1", "read_file", {"path": "a"})
    assert resp.stop_reason is StopReason.TOOL_USE
    assert (resp.usage.input_tokens, resp.usage.cache_read_tokens) == (20, 100)  # cached split out of input


def test_openai_incomplete_and_refusal():
    truncated = _openai_response(
        [{"id": "fc_1", "type": "function_call", "call_id": "c", "name": "read_file", "arguments": '{"pa'}],
        status="incomplete",
        incomplete="max_output_tokens",
    )
    resp = OpenAIProvider(client=SimpleNamespace(responses=_Responses(truncated))).complete(
        ModelRequest(model="m", messages=(Message.user("hi"),))
    )
    assert resp.stop_reason is StopReason.MAX_TOKENS
    assert resp.message.tool_uses[0].input == {}  # truncated JSON doesn't crash the provider

    refusal = _openai_response(
        [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "no"}],
            }
        ]
    )
    resp = OpenAIProvider(client=SimpleNamespace(responses=_Responses(refusal))).complete(
        ModelRequest(model="m", messages=(Message.user("hi"),))
    )
    assert resp.stop_reason is StopReason.CONTENT_FILTERED
    assert resp.message.text == "no"


def test_openai_streams_deltas_and_structured_output_format():
    response = _openai_response(
        [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Reading.", "annotations": []}],
            }
        ]
    )
    events = [
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="hmm", output_index=0),
        SimpleNamespace(type="response.output_text.delta", delta="Read", output_index=1),
        SimpleNamespace(type="response.output_text.delta", delta="ing.", output_index=1),
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"p', output_index=2),
    ]
    responses = _Responses(response, events)
    schema = {"type": "object"}
    req = ModelRequest(
        model="m", messages=(Message.user("hi"),), output_schema=schema, params={"text": {"verbosity": "low"}}
    )
    deltas = []
    resp = OpenAIProvider(client=SimpleNamespace(responses=responses)).complete(req, on_delta=deltas.append)
    assert [(d.kind, d.text, d.index) for d in deltas] == [
        ("reasoning", "hmm", 0),
        ("text", "Read", 1),
        ("text", "ing.", 1),
        ("tool_input", '{"p', 2),
    ]
    assert resp.message.text == "Reading." and resp.stop_reason is StopReason.END_TURN
    assert responses.calls[0]["stream"] is True
    assert responses.calls[0]["text"] == {
        "verbosity": "low",
        "format": {"type": "json_schema", "name": "output", "schema": schema, "strict": False},
    }


def test_openai_stop_cancels_the_stream():
    import threading

    stop = threading.Event()
    response = _openai_response([])

    class _Stopping(_EventStream):
        def __iter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta="Partial", output_index=0)
            stop.set()
            yield SimpleNamespace(type="response.output_text.delta", delta=" more", output_index=0)
            yield SimpleNamespace(type="response.completed", response=self.response)

    responses = _Responses(response)
    stream = _Stopping([], response)
    responses.create = lambda **kwargs: stream  # type: ignore[method-assign]
    with pytest.raises(ProviderCancelled) as info:
        OpenAIProvider(client=SimpleNamespace(responses=responses)).complete(
            ModelRequest(model="m", messages=(Message.user("hi"),)), stop=stop
        )
    assert stream.closed and info.value.partial is not None and info.value.partial.text == "Partial"
