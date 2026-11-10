"""Shell tool."""

from __future__ import annotations

import subprocess

from agent_core import ToolContext, tool


@tool(permission="ask")
def bash(ctx: ToolContext, command: str, timeout: int = 120) -> str:
    """Run a bash command in the working directory and return its combined
    stdout/stderr and exit code. Each call is a fresh shell, so cd and env vars
    do not persist between calls. Avoid interactive commands.

    Args:
        command: the bash command to run.
        timeout: seconds before the command is killed (max 600).
    """
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=ctx.cwd,
            capture_output=True,
            text=True,
            timeout=min(max(timeout, 1), 600),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return f"{out}\n[timed out after {timeout}s]"
    out = proc.stdout + (("\n" if proc.stdout and proc.stderr else "") + proc.stderr if proc.stderr else "")
    return f"{out.rstrip()}\n[exit code {proc.returncode}]".lstrip()
