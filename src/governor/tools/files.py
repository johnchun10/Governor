"""File tools: read, write, edit, list, grep."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess

from agent_core import ToolContext, ToolError, tool

SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".governor",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}
MAX_LINE = 2000


@tool(permission="allow")
def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int = 2000) -> str:
    """Read a text file. Returns numbered lines (like `cat -n`).

    Args:
        path: file path, relative to the working directory or absolute.
        offset: first line to return, 1-based.
        limit: maximum number of lines to return.
    """
    p = ctx.resolve(path)
    if not p.is_file():
        raise ToolError(f"{path} is not a file")
    data = p.read_bytes()
    if b"\0" in data[:8192]:
        raise ToolError(f"{path} looks like a binary file ({len(data)} bytes)")
    lines = data.decode("utf-8", errors="replace").splitlines()
    start = max(offset, 1) - 1
    chunk = lines[start : start + limit]
    out = [f"{i:6}\t{line[:MAX_LINE]}" for i, line in enumerate(chunk, start=start + 1)]
    if start + limit < len(lines):
        out.append(f"... ({len(lines) - start - limit} more lines; use offset={start + limit + 1})")
    return "\n".join(out) if out else "(empty file)"


@tool(permission="ask")
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Args:
        path: file path, relative to the working directory or absolute.
        content: the complete new file content.
    """
    p = ctx.resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content.encode())} bytes to {path}"


@tool(permission="ask")
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string in a file. old_string must match exactly (including
    whitespace) and be unique unless replace_all is true. Read the file first.

    Args:
        path: file path, relative to the working directory or absolute.
        old_string: exact text to replace.
        new_string: replacement text.
        replace_all: replace every occurrence instead of requiring a unique match.
    """
    p = ctx.resolve(path)
    if not p.is_file():
        raise ToolError(f"{path} is not a file")
    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if not old_string or count == 0:
        raise ToolError(f"old_string not found in {path}")
    if count > 1 and not replace_all:
        raise ToolError(f"old_string occurs {count} times in {path}; add context to make it unique or set replace_all")
    p.write_text(text.replace(old_string, new_string), encoding="utf-8")
    return f"edited {path} ({count if replace_all else 1} replacement{'s' if replace_all and count > 1 else ''})"


def _walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            yield os.path.join(dirpath, f)


@tool(permission="allow")
def list_files(ctx: ToolContext, pattern: str = "**/*", path: str = ".") -> str:
    """List files matching a glob pattern (e.g. "**/*.py", "src/*"). Skips .git,
    node_modules, virtualenvs and caches.

    Args:
        pattern: glob pattern relative to path.
        path: directory to search from.
    """
    root = ctx.resolve(path)
    if not root.is_dir():
        raise ToolError(f"{path} is not a directory")
    matches = []
    for full in _walk(str(root)):
        rel = os.path.relpath(full, root)
        if fnmatch.fnmatch(rel, pattern) or (pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])):
            matches.append(rel)
    if not matches:
        return "(no matches)"
    shown = matches[:500]
    more = f"\n... ({len(matches) - 500} more)" if len(matches) > 500 else ""
    return "\n".join(shown) + more


@tool(permission="allow")
def grep(ctx: ToolContext, pattern: str, path: str = ".", glob: str | None = None, ignore_case: bool = False) -> str:
    """Search file contents with a regular expression. Returns file:line:text matches.

    Args:
        pattern: regular expression to search for.
        path: file or directory to search.
        glob: only search files whose name matches this glob (e.g. "*.py").
        ignore_case: case-insensitive matching.
    """
    root = ctx.resolve(path)
    limit = 200
    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "--max-columns=300"]
        cmd += ["-i"] if ignore_case else []
        cmd += ["--glob", glob] if glob else []
        proc = subprocess.run([*cmd, "--", pattern, str(root)], capture_output=True, text=True, timeout=60)
        if proc.returncode > 1:
            raise ToolError(proc.stderr.strip())
        lines = proc.stdout.splitlines()
    else:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"bad regex: {exc}") from exc
        files = [str(root)] if root.is_file() else _walk(str(root))
        lines = []
        for f in files:
            if glob and not fnmatch.fnmatch(os.path.basename(f), glob):
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    for n, line in enumerate(fh, 1):
                        if rx.search(line):
                            lines.append(f"{f}:{n}:{line.rstrip()[:300]}")
            except (UnicodeDecodeError, OSError):
                continue
            if len(lines) > limit:
                break
    cwd = str(ctx.cwd) + os.sep
    lines = [ln.removeprefix(cwd) for ln in lines]
    if not lines:
        return "(no matches)"
    more = f"\n... ({len(lines) - limit} more matches)" if len(lines) > limit else ""
    return "\n".join(lines[:limit]) + more


FILE_TOOLS = [read_file, write_file, edit_file, list_files, grep]
