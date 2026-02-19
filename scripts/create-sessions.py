# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///
"""Generate session snapshots for marimo notebooks.

Runs marimo notebooks to completion and writes session snapshots to the
__marimo__/session/ directory next to each notebook file. This enables
instant loading of pre-computed outputs.

Each notebook is executed in its own uv sandbox, using the PEP 723
inline script metadata to resolve dependencies automatically.

Usage:
    # Single file
    uv run scripts/create-sessions.py notebook.py

    # Multiple files/folders
    uv run scripts/create-sessions.py notebooks/ examples/my_notebook.py

    # With CLI args passed to notebooks
    uv run scripts/create-sessions.py --cli-args '{"key": "value"}' notebooks/
    uv run scripts/create-sessions.py --argv '--flag value' notebooks/
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from marimo._server.files.directory_scanner import is_marimo_app
from marimo._utils.files import expand_file_patterns


def run_single(notebook_path: str, cli_args: dict, argv: list[str] | None) -> None:
    """Run a single notebook and write its session snapshot (in-process)."""
    from marimo._server.export import run_app_until_completion
    from marimo._server.file_router import AppFileRouter
    from marimo._server.utils import asyncio_run
    from marimo._session.state.serialize import (
        get_session_cache_file,
        serialize_session_view,
    )
    from marimo._utils.marimo_path import MarimoPath

    marimo_path = MarimoPath(notebook_path)
    file_router = AppFileRouter.from_filename(marimo_path)
    file_key = file_router.get_unique_file_key()
    assert file_key is not None
    file_manager = file_router.get_file_manager(file_key)

    session_view, did_error = asyncio_run(
        run_app_until_completion(file_manager, cli_args, argv)
    )

    if did_error:
        print("  Warning: notebook had errors during execution")

    cell_ids = list(file_manager.app.cell_manager.cell_ids())
    session_data = serialize_session_view(session_view, cell_ids)

    cache_file = get_session_cache_file(Path(notebook_path).resolve())
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(session_data, indent=2))

    status = "with errors" if did_error else "ok"
    print(f"  -> {cache_file} ({status})")

    if did_error:
        sys.exit(2)


def process_notebook_in_sandbox(
    notebook_path: Path, cli_args: dict, argv: list[str] | None
) -> tuple[bool, bool]:
    """Spawn a subprocess that runs this script with --single inside a uv sandbox.

    Uses marimo's own sandbox helpers to build the uv command, so the
    notebook's PEP 723 inline dependencies are resolved automatically.

    Returns (succeeded, had_errors).
    """
    from marimo._cli.sandbox import construct_uv_flags
    from marimo._utils.inline_script_metadata import PyProjectReader
    from marimo._utils.uv import find_uv_bin

    pyproject = PyProjectReader.from_filename(str(notebook_path))

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        uv_flags = construct_uv_flags(pyproject, tmp, [], [])
    atexit.register(lambda p=tmp_path: os.unlink(p))

    # Build: uv run <sandbox-flags> python <this-script> --single <notebook> ...
    uv_cmd = [find_uv_bin(), "run"] + uv_flags + [
        "python",
        __file__,
        "--single",
        str(notebook_path.resolve()),
        "--cli-args",
        json.dumps(cli_args),
    ]
    if argv is not None:
        uv_cmd.extend(["--argv", " ".join(argv)])

    result = subprocess.run(uv_cmd)

    if result.returncode == 0:
        return True, False
    elif result.returncode == 2:
        return True, True  # succeeded with errors
    else:
        return False, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate session snapshots for marimo notebooks.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more files, directories, or glob patterns",
    )
    parser.add_argument(
        "--cli-args",
        default="{}",
        help='JSON string of CLI args to pass to notebooks (default: "{}")',
    )
    parser.add_argument(
        "--argv",
        default=None,
        help="Space-separated argv to pass to notebooks (default: None)",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="(internal) Run a single notebook in-process and exit.",
    )
    args = parser.parse_args()

    # Parse CLI args
    try:
        cli_args: dict = json.loads(args.cli_args)  # type: ignore[type-arg]
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON for --cli-args: {e}", file=sys.stderr)
        sys.exit(1)

    argv: list[str] | None = None
    if args.argv is not None:
        argv = args.argv.split()

    # --single mode: run one notebook in-process (called from the sandbox subprocess)
    if args.single:
        run_single(args.single, cli_args, argv)
        return

    if not args.paths:
        parser.error("the following arguments are required: paths")

    # Discover files
    all_files = expand_file_patterns(tuple(args.paths))
    notebooks = [f for f in all_files if is_marimo_app(str(f))]

    if not notebooks:
        print("No marimo notebooks found in the given paths.")
        sys.exit(0)

    print(f"Found {len(notebooks)} marimo notebook(s)\n")

    succeeded = 0
    failed = 0

    for notebook_path in notebooks:
        print(f"Processing: {notebook_path}")
        ok, had_errors = process_notebook_in_sandbox(
            Path(notebook_path), cli_args, argv
        )
        if ok:
            succeeded += 1
        else:
            print(f"  Error: notebook processing failed", file=sys.stderr)
            failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
