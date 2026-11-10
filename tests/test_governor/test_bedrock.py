"""Converse request/response mapping, checked with botocore's Stubber (no network)."""

from __future__ import annotations

import boto3
import pytest
from botocore.stub import ANY, Stubber

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
from governor.providers.bedrock import BedrockProvider, collect_stream, to_converse

SPEC = ToolSpec(
    "read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
)


def client():
    return boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")


def test_request_mapping_with_cache_points_and_tool_results():
    reasoning = ReasoningBlock(
        raw={"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}}, provider="bedrock"
    )
    foreign = ReasoningBlock(raw={"type": "thinking"}, provider="mantle")
    req = ModelRequest(
        model="us.anthropic.claude-opus-5-5",
        system="be good",
        tools=(SPEC,),
        messages=(
            Message.user("read a.txt"),
            Message(
                "assistant", (reasoning, foreign, TextBlock(""), ToolUseBlock("t1", "read_file", {"path": "a.txt"}))
            ),
            Message("user", (ToolResultBlock("t1", "boom", is_error=True),)),
        ),
        params={"output_config": {"effort": "high"}},
    )
    k = to_converse(req)
    assert k["system"] == [{"text": "be good"}, {"cachePoint": {"type": "default"}}]
    assert k["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
    assert k["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"] == SPEC.input_schema
    assistant = k["messages"][1]["content"]
    assert assistant[0] == reasoning.raw  # passed back verbatim
    assert len(assistant) == 3  # foreign reasoning and blank text dropped; a cache point added
    assert k["messages"][2]["content"][0]["toolResult"]["status"] == "error"
    assert k["additionalModelRequestFields"] == {"output_config": {"effort": "high"}}
    assert "temperature" not in k["inferenceConfig"]


def test_cache_points_end_the_last_two_messages():
    point = {"cachePoint": {"type": "default"}}
    history = (
        Message.user("q1"),
        Message("assistant", (TextBlock("a1"),)),
        Message.user(" "),  # blank, so Converse gets no message for it
        Message.user("q2"),
    )
    k = to_converse(ModelRequest(model="m", system="s", tools=(SPEC,), messages=history))
    contents = [m["content"] for m in k["messages"]]
    assert contents == [[{"text": "q1"}], [{"text": "a1"}, point], [{"text": "q2"}, point]]
    assert sum(block == point for part in (k["system"], k["toolConfig"]["tools"], *contents) for block in part) == 4

    one = to_converse(ModelRequest(model="m", messages=(Message.user("hi"),)))
    assert one["messages"][0]["content"] == [{"text": "hi"}, point]
    plain = to_converse(ModelRequest(model="m", system="s", messages=history, cache=False))
    assert point not in plain["system"] and all(point not in m["content"] for m in plain["messages"])


def test_response_mapping_via_stubber():
    c = client()
    provider = BedrockProvider(client=c)
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "think", "signature": "s"}}},
                    {"text": "Reading."},
                    {"toolUse": {"toolUseId": "t9", "name": "read_file", "input": {"path": "a"}}},
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 12,
            "outputTokens": 7,
            "totalTokens": 19,
            "cacheReadInputTokens": 100,
            "cacheWriteInputTokens": 5,
        },
        "metrics": {"latencyMs": 42},
    }
    with Stubber(c) as stub:
        stub.add_response("converse", response, {"modelId": "m", "messages": ANY, "inferenceConfig": ANY})
        resp = provider.complete(ModelRequest(model="m", messages=(Message.user("hi"),), cache=False))
    assert resp.stop_reason is StopReason.TOOL_USE
    assert isinstance(resp.message.content[0], ReasoningBlock)
    assert resp.message.text == "Reading."
    assert resp.message.tool_uses == [ToolUseBlock("t9", "read_file", {"path": "a"})]
    assert (resp.usage.cache_read_tokens, resp.usage.cache_write_tokens) == (100, 5)


def test_error_hint_for_bare_model_id():
    c = client()
    with Stubber(c) as stub:
        stub.add_client_error(
            "converse",
            service_error_code="ValidationException",
            service_message="Invocation of model ID anthropic.x with on-demand throughput isn't supported.",
        )
        with pytest.raises(ProviderError) as info:
            BedrockProvider(client=c).complete(ModelRequest(model="anthropic.x", messages=(Message.user("hi"),)))
    assert "inference profile" in info.value.hint


def test_output_schema_becomes_converse_output_config():
    import json

    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    k = to_converse(ModelRequest(model="m", messages=(Message.user("hi"),), output_schema=schema))
    fmt = k["outputConfig"]["textFormat"]
    assert fmt["type"] == "json_schema"
    assert json.loads(fmt["structure"]["jsonSchema"]["schema"]) == schema
    assert "outputConfig" not in to_converse(ModelRequest(model="m", messages=(Message.user("hi"),)))


def _events():
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "think"}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "sig"}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "Read"}, "contentBlockIndex": 1}},
        {"contentBlockDelta": {"delta": {"text": "ing."}, "contentBlockIndex": 1}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t9", "name": "read_file"}}, "contentBlockIndex": 2}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"path": '}}, "contentBlockIndex": 2}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '"a"}'}}, "contentBlockIndex": 2}},
        {"contentBlockStop": {"contentBlockIndex": 2}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {"inputTokens": 12, "outputTokens": 7, "cacheReadInputTokens": 100},
                "metrics": {"latencyMs": 42},
            }
        },
    ]


def test_converse_stream_is_folded_into_one_response():
    deltas = []
    resp = collect_stream(_events(), "m", 0.0, deltas.append, None)
    assert [(d.kind, d.text, d.index) for d in deltas] == [
        ("reasoning", "think", 0),
        ("text", "Read", 1),
        ("text", "ing.", 1),
        ("tool_input", '{"path": ', 2),
        ("tool_input", '"a"}', 2),
    ]
    reasoning, text, call = resp.message.content
    assert reasoning == ReasoningBlock(
        raw={"reasoningContent": {"reasoningText": {"text": "think", "signature": "sig"}}}, provider="bedrock"
    )
    assert text == TextBlock("Reading.")
    assert call == ToolUseBlock("t9", "read_file", {"path": "a"})
    assert resp.stop_reason is StopReason.TOOL_USE
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cache_read_tokens) == (12, 7, 100)


def test_converse_stream_stop_and_malformed_tool_json():
    import threading

    stop = threading.Event()

    def events():
        for event in _events():
            if "messageStop" in event:
                stop.set()
            yield event

    with pytest.raises(ProviderCancelled) as info:
        collect_stream(events(), "m", 0.0, None, stop)
    assert info.value.partial is not None and info.value.partial.text == "Reading."

    broken = [
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "f"}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": "{not json"}}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    resp = collect_stream(broken, "m", 0.0, None, None)
    assert resp.message.tool_uses == [ToolUseBlock("t1", "f", {})]  # the schema check will say what is missing


def test_provider_streams_when_asked(monkeypatch: pytest.MonkeyPatch):
    class Client:
        def converse_stream(self, **kwargs):
            self.kwargs = kwargs
            return {"stream": _events()}

    client = Client()
    deltas = []
    resp = BedrockProvider(client=client).complete(
        ModelRequest(model="m", messages=(Message.user("hi"),)), on_delta=deltas.append
    )
    assert client.kwargs["modelId"] == "m" and len(deltas) == 5 and resp.message.text == "Reading."
