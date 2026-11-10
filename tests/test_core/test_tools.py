from __future__ import annotations

from typing import Literal

from agent_core import ToolContext, ToolRegistry, tool

from .conftest import noop, read_note, write_note


def test_schema_from_signature_and_docstring():
    @tool
    def sample(
        ctx: ToolContext, name: str, mode: Literal["a", "b"], count: int = 3, tags: list[str] | None = None
    ) -> str:
        """Do a thing.

        Args:
            name: who to greet.
            mode: which mode.
        """
        return ""

    assert sample.description == "Do a thing."
    assert sample.permission == "ask"
    s = sample.input_schema
    assert s["required"] == ["name", "mode"]
    assert s["additionalProperties"] is False
    assert s["properties"]["name"] == {"type": "string", "description": "who to greet."}
    assert s["properties"]["mode"]["enum"] == ["a", "b"]
    assert s["properties"]["count"] == {"type": "integer", "default": 3}
    assert s["properties"]["tags"]["type"] == "array"


def test_registry_name_patterns():
    reg = ToolRegistry().register(read_note, write_note, noop)
    assert reg.names(["*"]) == ["read_note", "write_note", "noop"]
    assert reg.names(["*", "-write_note"]) == ["read_note", "noop"]
    assert reg.names(["noop", "missing"]) == ["noop"]


def test_schema_validation_messages():
    from agent_core import parse_json, validate

    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "mode": {"enum": ["a", "b"]},
            "nested": {"$ref": "#/$defs/inner"},
            "either": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["n"],
        "additionalProperties": False,
        "$defs": {"inner": {"type": "object", "properties": {"x": {"type": "boolean"}}, "required": ["x"]}},
    }
    assert validate({"n": 2, "tags": ["a"], "mode": "a", "nested": {"x": True}, "either": None}, schema) == []
    assert validate({"n": True}, schema) == ["n: expected integer, got true"]
    assert validate({"n": 0}, schema) == ["n: must be >= 1"]
    problems = validate({"n": 1, "tags": [1, "b", "c"], "mode": "z", "nested": {}, "either": 2, "extra": 0}, schema)
    assert problems == [
        "tags: allows at most 2 items",
        "tags[0]: expected string, got 1",
        'mode: must be one of ["a", "b"], got "z"',
        "nested.x: missing",
        "either: does not match anyOf (none matched)",
        "extra: unexpected (known: n, tags, mode, nested, either)",
    ]
    assert validate(1.0, {"type": "integer"}) == [] and validate(1.5, {"type": "integer"})
    assert parse_json(' {"a": 1} ') == ({"a": 1}, None)
    assert parse_json("```json\n[1, 2]\n```") == ([1, 2], None)
    value, error = parse_json("{oops")
    assert value is None and error.startswith("not valid JSON")
