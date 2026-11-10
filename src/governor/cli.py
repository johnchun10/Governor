"""Entry point: ``govern`` opens the terminal UI in the current directory.

There are no flags and no other subcommands: everything is a slash command in
the TUI, and nothing runs outside it. Code that needs a run
without the UI builds a ``Session`` with ``governor.build_session`` (README,
"Library use").
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import load_environment

USAGE = "usage: govern    open the terminal UI in the current directory\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        helping = argv[0] in ("-h", "--help", "help")
        print(USAGE, end="", file=sys.stdout if helping else sys.stderr)
        return 0 if helping else 2
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("govern needs a terminal", file=sys.stderr)
        return 2
    load_environment(Path.cwd())
    from .tui import GovernorApp

    app = GovernorApp(Path.cwd())
    app.run()
    app.session.close()
    if app.session.trace:
        print(f"trace: {app.session.trace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
