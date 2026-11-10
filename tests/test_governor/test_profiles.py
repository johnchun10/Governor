from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_core import AgentSpec
from governor import profiles


def test_profile_defaults_come_from_spec() -> None:
    assert profiles.parse_profile("---\nname: x\n---\nPrompt\n") == AgentSpec(name="x", system_prompt="Prompt")


def test_profile_values_follow_type_hints() -> None:
    spec = profiles.parse_profile(
        "---\n"
        "name: x\n"
        "tools: read_file\n"
        "max_turns: '12'\n"
        "max_tokens: '2048'\n"
        "max_invalid_calls: '2'\n"
        "subagent: false\n"
        "permissions: {bash: deny}\n"
        "params: {temperature: 0.5}\n"
        "output_schema: {}\n"
        "---\nPrompt\n"
    )
    assert spec.tools == ["read_file"]
    assert (spec.max_turns, spec.max_tokens, spec.max_invalid_calls) == (12, 2048, 2)
    assert spec.subagent is False
    assert spec.permissions == {"bash": "deny"}
    assert spec.params == {"temperature": 0.5}
    assert spec.output_schema == {}


@pytest.mark.parametrize("field", ["model", "output_schema"])
def test_optional_profile_fields_keep_none(field: str) -> None:
    spec = profiles.parse_profile(f"---\nname: x\n{field}: null\n---\n")
    assert getattr(spec, field) is None


@pytest.mark.parametrize("field", ["permissions", "params"])
@pytest.mark.parametrize("value", ["{}", "null"])
def test_mapping_profile_fields(field: str, value: str) -> None:
    spec = profiles.parse_profile(f"---\nname: x\n{field}: {value}\n---\n")
    assert getattr(spec, field) == {}


def test_unknown_profile_key_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"custom\.md: unknown frontmatter keys: max_truns"):
        profiles.parse_profile("---\nname: x\nmax_truns: 10\n---\n", "custom.md")


@dataclass
class ExtendedSpec(AgentSpec):
    temperature: float | None = 0.5


@pytest.mark.parametrize("value, expected", [("'0.75'", 0.75), ("null", None)])
def test_new_spec_field_needs_no_parser_change(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: float | None
) -> None:
    monkeypatch.setattr(profiles, "AgentSpec", ExtendedSpec)
    spec = profiles.parse_profile(f"---\nname: x\ntemperature: {value}\n---\n")
    assert isinstance(spec, ExtendedSpec)
    assert spec.temperature == expected
    assert profiles.parse_profile("---\nname: x\n---\n").temperature == 0.5


@pytest.mark.parametrize("front", ["[]", "false", "42"])
def test_frontmatter_requires_mapping(front: str) -> None:
    with pytest.raises(ValueError, match="frontmatter must be a mapping"):
        profiles.parse_profile(f"---\n{front}\n---\n")


def test_invalid_profile_value_reports_source_and_field() -> None:
    with pytest.raises(ValueError, match=r"custom\.md: invalid max_turns"):
        profiles.parse_profile("---\nname: x\nmax_turns: nope\n---\n", "custom.md")


def test_system_prompt_is_the_body_never_a_frontmatter_key() -> None:
    with pytest.raises(ValueError, match="system_prompt is the text after the frontmatter"):
        profiles.parse_profile("---\nname: x\nsystem_prompt: hidden\n---\nBody\n")


def test_a_bare_text_field_keeps_its_default_but_a_bare_name_or_list_does_not() -> None:
    assert profiles.parse_profile("---\nname: x\ndescription:\n---\nP\n").description == AgentSpec(name="x").description
    with pytest.raises(ValueError, match="needs a name"):
        profiles.parse_profile("---\nname:\n---\nP\n")
    with pytest.raises(ValueError, match="invalid tools"):
        profiles.parse_profile("---\nname: x\ntools:\n---\nP\n")


@pytest.mark.parametrize(("value", "expected"), [('"false"', False), ("false", False), ('"True"', True)])
def test_quoted_booleans_mean_what_they_say(value: str, expected: bool) -> None:
    assert profiles.parse_profile(f"---\nname: x\nsubagent: {value}\n---\nP\n").subagent is expected
