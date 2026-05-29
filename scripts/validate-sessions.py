# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.8",
# ]
# ///
"""Validate session JSON files for marimo notebooks.

Checks that:
1. Every marimo notebook has a corresponding session JSON file.
2. Session JSON files don't contain error patterns (e.g. ModuleNotFoundError).
3. Session JSON files are up-to-date with their notebook code (code hashes match).

Usage:
    uv run scripts/validate-sessions.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

ERROR_PATTERNS = ["ModuleNotFoundError"]


def is_ci() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def report_error(file: str, msg: str) -> None:
    if is_ci():
        print(f"::error file={file}::{msg}")
    else:
        print(f"  ERROR: {msg}")


def get_session_path(notebook_path: Path) -> Path:
    return notebook_path.parent / "__marimo__" / "session" / f"{notebook_path.name}.json"


def find_notebooks() -> list[Path]:
    """Find marimo notebooks under notebooks/, ignoring generated __marimo__ dirs.

    Uses the same heuristic as the check-notebooks CI workflow: a .py file is a
    marimo notebook if it imports marimo and constructs an app.
    """
    notebooks = []
    for f in sorted(NOTEBOOKS_DIR.rglob("*.py")):
        if "__marimo__" in f.parts:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if "import marimo" in text and "marimo.App" in text:
            notebooks.append(f)
    return notebooks


def check_session_exists(notebooks: list[Path]) -> bool:
    """Check every notebook has a session JSON. Returns True if all pass."""
    print("Checking that every notebook has a session JSON...")
    ok = True
    for nb in notebooks:
        session = get_session_path(nb)
        if not session.exists():
            rel = nb.relative_to(REPO_ROOT)
            report_error(str(rel), f"Missing session JSON: {session.relative_to(REPO_ROOT)}")
            ok = False
    if ok:
        print("  All notebooks have session JSON files.")
    return ok


def check_error_patterns(notebooks: list[Path]) -> bool:
    """Check session JSONs for known error patterns. Returns True if all pass."""
    print("Checking session JSONs for errors...")
    ok = True
    for nb in notebooks:
        session = get_session_path(nb)
        if not session.exists():
            continue
        content = session.read_text()
        rel_nb = nb.relative_to(REPO_ROOT)
        rel_session = session.relative_to(REPO_ROOT)
        for pattern in ERROR_PATTERNS:
            if pattern in content:
                report_error(
                    str(rel_nb),
                    f"'{rel_nb}' has '{pattern}' in its session JSON ({rel_session})",
                )
                ok = False
    if ok:
        print("  All session JSON files are clean.")
    return ok


def check_session_freshness(notebooks: list[Path]) -> bool:
    """Check session JSONs are up-to-date with notebook code.

    Matches notebook cells to session cells by code_hash rather than by
    position or cell ID, since cells may be ordered differently. The notebook
    code hashes are computed via marimo's own session-cache helper (the same
    one `marimo export session` uses to decide whether a snapshot is stale), so
    the hashing stays in lockstep with how snapshots are written.

    Returns True if all pass.
    """
    from collections import Counter

    from marimo._server.export._session_cache import current_notebook_code_hashes
    from marimo._utils.marimo_path import MarimoPath

    print("Checking session JSONs are up-to-date...")
    ok = True
    for nb in notebooks:
        session_path = get_session_path(nb)
        if not session_path.exists():
            continue

        rel_nb = nb.relative_to(REPO_ROOT)

        # Compute the notebook's current per-cell code hashes.
        try:
            notebook_hashes = Counter(
                current_notebook_code_hashes(MarimoPath(str(nb)))
            )
        except Exception as e:
            report_error(str(rel_nb), f"Could not parse notebook '{rel_nb}': {e}")
            ok = False
            continue

        # Load the session JSON's stored code hashes.
        session_data = json.loads(session_path.read_text())
        session_hashes = Counter(
            c.get("code_hash") for c in session_data.get("cells", [])
        )

        # Compare as multisets — order doesn't matter.
        if notebook_hashes != session_hashes:
            only_in_notebook = notebook_hashes - session_hashes
            only_in_session = session_hashes - notebook_hashes
            changed = len(only_in_notebook) + len(only_in_session)
            report_error(
                str(rel_nb),
                f"'{rel_nb}' has {changed} cell(s) out of sync with its session JSON — re-run create-sessions.py",
            )
            ok = False

    if ok:
        print("  All session JSON files are up-to-date.")
    return ok


def main() -> None:
    notebooks = find_notebooks()
    if not notebooks:
        print("No marimo notebooks found.")
        sys.exit(0)

    print(f"Found {len(notebooks)} marimo notebook(s)\n")

    results = [
        check_session_exists(notebooks),
        check_error_patterns(notebooks),
        check_session_freshness(notebooks),
    ]

    print()
    if all(results):
        print("All checks passed.")
    else:
        print("Some checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
