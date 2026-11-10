"""External tools: any executable, in any language, described by a JSON manifest.

Drop ``<name>.json`` into ``.governor/tools/``:

    {
      "name": "word_count",
      "description": "Count words in a text.",
      "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
      "command": ["node", "tools/word_count.js"],
      "permission": "allow",
      "timeout": 60
    }

Governor runs ``command`` in the working directory with the tool input as
JSON on stdin. Whatever the command prints on stdout becomes the result. If
stdout is a JSON object with an ``error`` key, the result is marked as an
error. A non-zero exit code is also reported as an error.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from agent_core import Tool, ToolContext, ToolError


def _make_runner(manifest: dict[str, Any]) -> Any:
    command = manifest["command"]
    timeout = int(manifest.get("timeout", 60))

    def run(ctx: ToolContext, **args: Any) -> str:
        try:
            proc = subprocess.run(
                command,
                cwd=ctx.cwd,
                input=json.dumps(args),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"cannot run {command[0]!r}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"timed out after {timeout}s") from exc
        out = proc.stdout.strip()
        if proc.returncode != 0:
            raise ToolError(f"exit code {proc.returncode}: {(proc.stderr or out).strip()[:2000]}")
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            return out
        if isinstance(parsed, dict) and "error" in parsed:
            raise ToolError(str(parsed["error"]))
        if isinstance(parsed, dict) and "content" in parsed:
            return str(parsed["content"])
        return out

    return run


def load_external_tools(directory: Path) -> list[Tool]:
    tools = []
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        manifest = json.loads(path.read_text())
        tools.append(
            Tool(
                name=manifest["name"],
                description=manifest.get("description", ""),
                input_schema=manifest.get("input_schema", {"type": "object", "properties": {}}),
                fn=_make_runner(manifest),
                permission=manifest.get("permission", "ask"),
            )
        )
    return tools
