# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.8",
# ]
# ///
"""Generate session snapshots for marimo notebooks.

Thin wrapper around `marimo export session`, the supported CLI for executing
notebooks and writing session snapshots to `__marimo__/session/<notebook>.py.json`
next to each notebook. These snapshots enable instant loading of pre-computed
outputs in the gallery.

Each notebook runs in its own uv sandbox (`--sandbox`, i.e. `uv run --isolated`),
so its PEP 723 inline dependencies are resolved automatically.

Usage:
    # Single file
    uv run scripts/create-sessions.py notebook.py

    # Multiple files or directories
    uv run scripts/create-sessions.py notebooks/ examples/my_notebook.py

    # Forward extra args to the notebook(s), parsed via mo.cli_args()
    uv run scripts/create-sessions.py notebook.py -- --key value
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def export_session(path: str, notebook_args: list[str]) -> int:
    """Run `marimo export session --sandbox` for a single file or directory."""
    cmd = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "session",
        "--sandbox",
        # Always (re)generate the requested snapshots; freshness gating lives
        # in validate-sessions.py.
        "--force-overwrite",
        path,
        *notebook_args,
    ]
    return subprocess.run(cmd).returncode


def main() -> None:
    argv = sys.argv[1:]

    # Everything after a literal "--" is forwarded to the notebook(s) as argv.
    notebook_args: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        notebook_args = argv[sep + 1 :]
        argv = argv[:sep]

    parser = argparse.ArgumentParser(
        description="Generate session snapshots for marimo notebooks.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more notebook files or directories",
    )
    args = parser.parse_args(argv)

    failed = 0
    for path in args.paths:
        print(f"Processing: {path}")
        if export_session(path, notebook_args) != 0:
            print(f"  Error: failed to export session for {path}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"\n{failed} path(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
