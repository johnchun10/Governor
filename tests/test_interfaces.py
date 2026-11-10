"""README.md's "Interfaces" section lists every public class (fields, public methods) and module-level
function under ``src/``, generated from the source so it cannot drift.

``pytest`` fails when the section no longer matches the code; ``python tests/test_interfaces.py``
rewrites it (the text between the two ``interfaces`` markers in README.md).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- interfaces:start -->", "<!-- interfaces:end -->"
PACKAGES = ("agent_core", "governor")


def signature(fn: ast.FunctionDef) -> str:
    a = fn.args
    parts: list[str] = []
    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    for arg, default in zip(positional, defaults, strict=True):
        if arg.arg not in ("self", "cls"):
            parts.append(arg.arg + (f"={ast.unparse(default)}" if default is not None else ""))
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        parts.append(arg.arg + (f"={ast.unparse(default)}" if default is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    returns = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None and fn.name != "__init__" else ""
    return f"{fn.name}({', '.join(parts)}){returns}"


def describe_class(node: ast.ClassDef) -> str:
    decorators = [ast.unparse(d) for d in node.decorator_list]
    bases = [ast.unparse(b) for b in node.bases]
    kind = (
        "protocol" if "Protocol" in bases else "dataclass" if any(d.startswith("dataclass") for d in decorators) else ""
    )
    fields = [
        n.target.id
        for n in node.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and not n.target.id.startswith("_")
    ]
    methods = [
        signature(n)
        for n in node.body
        if isinstance(n, ast.FunctionDef) and (not n.name.startswith("_") or n.name == "__init__")
    ]
    head = f"`{node.name}`"
    if kind:
        head += f" ({kind})"
    elif bases:
        head += f" · {', '.join(bases)}"
    parts = ([f"fields {', '.join(fields)}"] if fields else []) + [f"`{m}`" for m in methods]
    return f"- {head}: {' · '.join(parts)}" if parts else f"- {head}"


def describe_module(path: Path) -> list[str]:
    lines = []
    for node in ast.parse(path.read_text(), str(path)).body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            lines.append(describe_class(node))
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            lines.append(f"- `{signature(node)}`")
    return lines


def interfaces() -> str:
    """The section body: one block per module that has a public class or function."""
    out: list[str] = []
    for package in PACKAGES:
        for path in sorted((ROOT / "src" / package).rglob("*.py")):
            lines = describe_module(path)
            if lines:
                out += [f"**`{path.relative_to(ROOT)}`**", "", *lines, ""]
    return "\n".join(out).rstrip() + "\n"


def section(readme: str) -> str:
    start, end = readme.index(START) + len(START), readme.index(END)
    return readme[start:end].strip("\n") + "\n"


def test_readme_lists_the_public_interfaces():
    readme = README.read_text()
    assert START in readme and END in readme, "README.md lost its interfaces markers"
    assert section(readme) == interfaces(), (
        "README.md's Interfaces section is stale: run `python tests/test_interfaces.py`"
    )


if __name__ == "__main__":
    text = README.read_text()
    if START not in text:
        sys.exit(f"add {START} and {END} to README.md first")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{START}\n\n{interfaces()}\n{END}{tail}")
    print(f"README.md: Interfaces section rewritten ({interfaces().count(chr(10))} lines)")
