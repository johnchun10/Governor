from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_core import Agent, AgentSpec, ToolError
from agent_core.testing import text, tool_call
from governor import Permissions, load_profiles, parse_profile
from governor.tools.external import load_external_tools
from governor.tools.files import edit_file, grep, list_files, read_file


def test_edit_file_requires_unique_match(ctx, tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("a = 1\na = 1\n")
    with pytest.raises(ToolError, match="occurs 2 times"):
        edit_file.fn(ctx, "x.py", "a = 1", "a = 2")
    edit_file.fn(ctx, "x.py", "a = 1", "a = 2", replace_all=True)
    assert f.read_text() == "a = 2\na = 2\n"
    with pytest.raises(ToolError, match="not found"):
        edit_file.fn(ctx, "x.py", "zzz", "y")


def test_read_list_grep(ctx, tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("def foo():\n    return 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "skip.py").write_text("def foo(): pass\n")
    assert "     2\t    return 1" in read_file.fn(ctx, "src/m.py")
    assert list_files.fn(ctx, "**/*.py") == "src/m.py"
    out = grep.fn(ctx, r"def foo")
    assert "src/m.py:1:" in out and ".git" not in out


def test_external_tool_roundtrip(ctx, tmp_path: Path):
    tools_dir = tmp_path / ".governor" / "tools"
    tools_dir.mkdir(parents=True)
    script = tmp_path / "upper.py"
    script.write_text(
        "import json,sys\nargs=json.load(sys.stdin)\nprint(json.dumps({'content': args['text'].upper()}))\n"
    )
    (tools_dir / "upper.json").write_text(
        json.dumps(
            {
                "name": "upper",
                "description": "Uppercase text.",
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                "command": [sys.executable, str(script)],
                "permission": "allow",
            }
        )
    )
    [t] = load_external_tools(tools_dir)
    assert t.name == "upper" and t.permission == "allow"
    assert t.fn(ctx, text="hi") == "HI"


def test_profile_parsing():
    spec = parse_profile("---\nname: x\nmodel: haiku\ntools: [read_file]\nsubagent: false\n---\nBe brief.\n")
    assert (spec.name, spec.model, spec.tools, spec.subagent, spec.system_prompt) == (
        "x",
        "haiku",
        ["read_file"],
        False,
        "Be brief.",
    )
    builtins = load_profiles()
    assert {"general", "explore", "planner"} <= set(builtins)
    assert isinstance(builtins["explore"], AgentSpec)


def test_interactive_permissions_asker():
    asked = []

    def asker(tool, args):
        asked.append(tool)
        return "a"  # always allow from now on

    perms = Permissions(asker=asker)
    assert perms.check("bash", {"command": "ls"}, "ask") is None
    assert perms.check("bash", {"command": "ls"}, "ask") is None
    assert asked == ["bash"]  # remembered after 'a'
    assert "declined" in Permissions(asker=lambda t, a: "n").check("bash", {}, "ask")
    assert Permissions(asker=None).check("read_file", {}, "allow") is None


def test_real_tools_through_core_loop(runtime, tmp_path: Path):
    rt, provider, _ = runtime
    provider._script += [tool_call("write_file", {"path": "out.txt", "content": "hi"}), text("done")]
    result = Agent(rt.profiles["general"], rt, cwd=tmp_path).run("write out.txt")
    assert result.text == "done" and (tmp_path / "out.txt").read_text() == "hi"
