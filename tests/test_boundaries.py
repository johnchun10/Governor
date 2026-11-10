"""Enforce the package and layer boundaries.

1. agent_core imports only the standard library and itself (no boto3, yaml or governor).
2. governor imports agent_core only through its public API (``from agent_core import X``),
   never ``agent_core.<module>``, so core can reorganize its internals freely.
   ``agent_core.testing`` is the one exception, for test helpers.
3. The application layer is UI-free: no module under governor but ``tui.py`` imports
   ``textual`` or ``rich``, and no module but ``cli.py`` imports ``governor.tui``
   (AGENTS.md "Architecture and abstraction", rules 5 and 7).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED_CORE_SUBMODULES = {"agent_core.testing"}
UI_TOOLKITS = {"textual", "rich"}
SURFACES = {"governor.tui", "governor.cli"}


def imports(path: Path) -> list[tuple[str, int]]:
    """Absolute module names imported by ``path``; relative imports are resolved within ``src``."""
    package = path.relative_to(SRC).parent.parts  # ("governor", "modes") for governor/modes/qa.py
    out = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    out.append((node.module, node.lineno))
            else:
                base = package[: len(package) - (node.level - 1)]
                out.append((".".join([*base, *([node.module] if node.module else [])]), node.lineno))
    return out


def governor_modules(*except_: str) -> list[Path]:
    return [p for p in (SRC / "governor").rglob("*.py") if p.name not in except_]


def test_core_depends_only_on_stdlib():
    allowed = set(sys.stdlib_module_names) | {"agent_core", "__future__"}
    bad = [
        f"{p.relative_to(SRC)}:{line} imports {mod}"
        for p in (SRC / "agent_core").rglob("*.py")
        for mod, line in imports(p)
        if mod.split(".")[0] not in allowed
    ]
    assert not bad, "agent_core must stay stdlib-only:\n" + "\n".join(bad)


def test_governor_uses_core_public_api_only():
    bad = [
        f"{p.relative_to(SRC)}:{line} imports {mod}"
        for p in governor_modules()
        for mod, line in imports(p)
        if mod.startswith("agent_core.") and mod not in ALLOWED_CORE_SUBMODULES
    ]
    assert not bad, "import from `agent_core` (its __init__), not its submodules:\n" + "\n".join(bad)


def test_only_the_tui_imports_a_ui_toolkit():
    bad = [
        f"{p.relative_to(SRC)}:{line} imports {mod}"
        for p in governor_modules("tui.py")
        for mod, line in imports(p)
        if mod.split(".")[0] in UI_TOOLKITS
    ]
    assert not bad, "only tui.py may import textual or rich; results are data, the surface renders them:\n" + "\n".join(
        bad
    )


def test_application_layer_never_imports_a_surface():
    bad = [
        f"{p.relative_to(SRC)}:{line} imports {mod}"
        for p in governor_modules("tui.py", "cli.py")
        for mod, line in imports(p)
        if mod in SURFACES
    ]
    assert not bad, "application code talks to the human through ports.Interaction, never a surface:\n" + "\n".join(bad)
