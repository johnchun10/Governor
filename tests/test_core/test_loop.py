from __future__ import annotations

import threading
import time

from agent_core import Agent, AgentSpec, Message, ModelRequest, StaticPolicy, ToolResultBlock, Usage
from agent_core.testing import ScriptedProvider, text, tool_call, tool_calls

from .conftest import NOTES


def results_in(request: ModelRequest) -> list[ToolResultBlock]:
    return [b for b in request.messages[-1].content if isinstance(b, ToolResultBlock)]


def test_tool_call_then_final_answer(make_runtime):
    NOTES["hello"] = "hi there"
    provider = ScriptedProvider([tool_call("read_note", {"name": "hello"}), text("It says hi there.")])
    runtime, events = make_runtime(provider)
    result = Agent(runtime.profiles["general"], runtime).run("what does hello say?")

    assert (result.stop, result.text, result.turns) == ("end_turn", "It says hi there.", 2)
    [res] = results_in(provider.requests[1])
    assert not res.is_error and res.content == "hi there"
    assert [e.type for e in events.events] == [
        "run_start",
        "model_call",
        "tool_call",
        "tool_result",
        "delta",
        "delta",
        "model_call",
        "run_end",
    ]
    assert "".join(e.data["text"] for e in events.of_type("delta")) == "It says hi there."
    assert result.usage.input_tokens == 20  # two calls x 10
    assert result.usage.cost_usd > 0  # priced by the injected ModelInfo


def test_tool_error_is_returned_not_raised(make_runtime):
    provider = ScriptedProvider([tool_call("read_note", {"name": "missing"}), text("not found")])
    runtime, _ = make_runtime(provider)
    result = Agent(runtime.profiles["general"], runtime).run("read it")
    [res] = results_in(provider.requests[1])
    assert res.is_error and "no note named" in res.content
    assert result.stop == "end_turn"


def test_unknown_tool_and_missing_args(make_runtime):
    provider = ScriptedProvider([tool_calls([("nope", {}), ("read_note", {})]), text("ok")])
    runtime, _ = make_runtime(provider)
    Agent(runtime.profiles["general"], runtime).run("x")
    unknown, missing = results_in(provider.requests[1])
    assert unknown.is_error and "unknown tool" in unknown.content
    assert missing.is_error and "name: missing" in missing.content


def test_permission_refusal_reaches_model(make_runtime):
    provider = ScriptedProvider([tool_call("write_note", {"name": "a", "text": "x"}), text("denied")])
    runtime, _ = make_runtime(provider, policy=StaticPolicy())  # "ask" without yolo -> refused
    Agent(runtime.profiles["general"], runtime).run("write")
    [res] = results_in(provider.requests[1])
    assert res.is_error and "permission denied" in res.content
    assert "a" not in NOTES


def test_profile_tool_subset_is_enforced(make_runtime):
    spec = AgentSpec("ro", tools=["read_note"])
    provider = ScriptedProvider([tool_call("write_note", {"name": "a", "text": "x"}), text("ok")])
    runtime, _ = make_runtime(provider, profiles={"ro": spec})
    Agent(spec, runtime).run("x")
    assert [t.name for t in provider.requests[0].tools] == ["read_note"]
    [res] = results_in(provider.requests[1])
    assert res.is_error and "unknown tool" in res.content


def test_max_turns(make_runtime):
    spec = AgentSpec("general", max_turns=3)
    provider = ScriptedProvider([tool_call("noop") for _ in range(3)])
    runtime, _ = make_runtime(provider, profiles={"general": spec})
    result = Agent(spec, runtime).run("loop forever")
    assert result.stop == "max_turns" and result.turns == 3


def test_parallel_subagents_nest_and_roll_up_usage(make_runtime):
    gate = threading.Barrier(2, timeout=5)  # both children must be running at once

    def router(req: ModelRequest):
        first = req.messages[0].text
        if first == "top task":
            if len(req.messages) == 1:
                return tool_calls(
                    [("task", {"agent": "explore", "prompt": "A"}), ("task", {"agent": "explore", "prompt": "B"})]
                )
            return text("combined: " + " | ".join(r.content for r in results_in(req)))
        gate.wait()
        return text(f"report {first}", usage=Usage(100, 50))

    provider = ScriptedProvider(router=router)
    profiles = {"general": AgentSpec("general"), "explore": AgentSpec("explore", "reader", tools=["read_note"])}
    runtime, events = make_runtime(provider, profiles=profiles)
    started = time.monotonic()
    result = Agent(profiles["general"], runtime).run("top task")

    assert time.monotonic() - started < 5
    assert result.text == "combined: report A | report B"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (220, 110)
    starts = events.of_type("run_start")
    children = [e for e in starts if e.depth == 1]
    assert len(children) == 2 and all(c.parent_run_id == starts[0].run_id for c in children)


def test_depth_limit_hides_task_tool(make_runtime):
    spec = AgentSpec("general")
    provider = ScriptedProvider([text("child at depth limit sees no task tool")])
    runtime, _ = make_runtime(provider, profiles={"general": spec}, max_depth=1)
    Agent(spec, runtime, depth=1).run("x")
    assert "task" not in [t.name for t in provider.requests[0].tools]


def test_model_override_applies_to_whole_tree(make_runtime):
    provider = ScriptedProvider([text("ok")])
    runtime, _ = make_runtime(provider, model_override="cheap")
    Agent(AgentSpec("general", model="expensive"), runtime).run("x")
    assert provider.requests[0].model == "fake-cheap"


def test_chat_history_is_append_only(make_runtime):
    provider = ScriptedProvider([text("one"), text("two")])
    runtime, _ = make_runtime(provider)
    agent = Agent(runtime.profiles["general"], runtime)
    r1 = agent.run("first")
    r2 = agent.run("second", history=r1.messages)
    assert r2.messages[: len(r1.messages)] == r1.messages
    assert [m.text for m in provider.requests[1].messages] == ["first", "one", "second"]


class Recall:
    def system(self, run):
        return "STATIC"

    def before_turn(self, run):
        return f"memory for turn {run.turn}"


def test_context_provider_hooks(make_runtime):
    provider = ScriptedProvider([tool_call("noop"), text("done")])
    runtime, events = make_runtime(provider, context_providers=[Recall()])
    Agent(runtime.profiles["general"], runtime).run("task")
    first, second = provider.requests
    assert first.system.endswith("STATIC")
    assert "memory for turn 1" in first.messages[-1].content[-1].text
    assert "memory for turn 2" in second.messages[-1].content[-1].text
    assert second.messages[0] == first.messages[0]  # earlier messages never edited
    assert len(events.of_type("context")) == 2


def test_wrong_argument_types_and_extras_are_reported(make_runtime):
    provider = ScriptedProvider([tool_call("write_note", {"name": 3, "text": "x", "bogus": 1}), text("ok")])
    runtime, events = make_runtime(provider)
    Agent(runtime.profiles["general"], runtime).run("x")
    [res] = results_in(provider.requests[1])
    assert res.is_error
    assert "name: expected string, got 3" in res.content and "bogus: unexpected" in res.content
    assert events.of_type("tool_result")[0].data["outcome"] == "invalid"
    assert "a" not in NOTES


def test_repeated_invalid_calls_end_the_run(make_runtime):
    spec = AgentSpec("general", max_invalid_calls=2)
    provider = ScriptedProvider([tool_call("read_note", {}), tool_call("read_note", {}), text("never sent")])
    runtime, _ = make_runtime(provider, profiles={"general": spec})
    result = Agent(spec, runtime).run("x")
    assert result.stop == "invalid_tool_calls" and result.turns == 2
    assert len(provider.requests) == 2

    # A valid call in between resets the streak.
    NOTES["n"] = "v"
    provider = ScriptedProvider(
        [tool_call("read_note", {}), tool_call("read_note", {"name": "n"}), tool_call("read_note", {}), text("done")]
    )
    runtime, _ = make_runtime(provider, profiles={"general": spec})
    assert Agent(spec, runtime).run("x").stop == "end_turn"


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "why": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_structured_output_is_parsed_and_validated(make_runtime):
    spec = AgentSpec("json", output_schema=SCHEMA)
    provider = ScriptedProvider([text('```json\n{"answer": 4, "why": "math"}\n```')])
    runtime, _ = make_runtime(provider, profiles={"json": spec})
    result = Agent(spec, runtime).run("2+2?")
    assert result.stop == "end_turn" and result.structured == {"answer": 4, "why": "math"}
    assert provider.requests[0].output_schema == SCHEMA


def test_structured_output_gets_one_corrective_turn(make_runtime):
    spec = AgentSpec("json", output_schema=SCHEMA)
    provider = ScriptedProvider([text('{"answer": "four"}'), text('{"answer": 4}')])
    runtime, _ = make_runtime(provider, profiles={"json": spec})
    result = Agent(spec, runtime).run("2+2?")
    assert result.stop == "end_turn" and result.structured == {"answer": 4}
    fix = provider.requests[1].messages[-1].text
    assert "answer: expected integer" in fix and "corrected JSON" in fix

    provider = ScriptedProvider([text("not json"), text("still not json")])
    runtime, _ = make_runtime(provider, profiles={"json": spec})
    result = Agent(spec, runtime).run("2+2?")
    assert result.stop == "invalid_output" and result.structured is None and result.turns == 2


def test_stop_event_cancels_the_run(make_runtime):
    stop = threading.Event()
    provider = ScriptedProvider([tool_call("noop"), text("never")])
    runtime, events = make_runtime(provider, stop=stop)
    stop.set()
    result = Agent(runtime.profiles["general"], runtime).run("x")
    assert result.stop == "cancelled" and len(provider.requests) == 0
    assert events.of_type("run_end")[0].data["stop"] == "cancelled"

    # Set while the model call is in flight: the provider raises ProviderCancelled.
    stop = threading.Event()

    def slow(req):
        stop.set()
        return tool_call("noop")

    provider = ScriptedProvider([slow])
    runtime, _ = make_runtime(provider, stop=stop)
    result = Agent(runtime.profiles["general"], runtime).run("x")
    assert result.stop == "cancelled" and result.messages == [Message.user("x")]
