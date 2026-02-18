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

Usage:
    # Single file
    python generate_snapshots.py notebook.py

    # Multiple files/folders
    python generate_snapshots.py notebooks/ examples/my_notebook.py

    # With CLI args passed to notebooks
    python generate_snapshots.py --cli-args '{"key": "value"}' notebooks/
    python generate_snapshots.py --argv '--flag value' notebooks/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from marimo._server.export import run_app_until_completion
from marimo._server.file_router import AppFileRouter
from marimo._server.files.directory_scanner import is_marimo_app
from marimo._server.utils import asyncio_run
from marimo._session.state.serialize import (
    get_session_cache_file,
    serialize_session_view,
)
from marimo._utils.files import expand_file_patterns
from marimo._utils.marimo_path import MarimoPath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate session snapshots for marimo notebooks.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
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
        try:
            # Create file manager
            marimo_path = MarimoPath(str(notebook_path))
            file_router = AppFileRouter.from_filename(marimo_path)
            file_key = file_router.get_unique_file_key()
            assert file_key is not None
            file_manager = file_router.get_file_manager(file_key)

            # Run notebook to completion
            session_view, did_error = asyncio_run(
                run_app_until_completion(file_manager, cli_args, argv)
            )

            if did_error:
                print(f"  Warning: notebook had errors during execution")

            # Get cell IDs in document order
            cell_ids = list(file_manager.app.cell_manager.cell_ids())

            # Serialize session view
            session_data = serialize_session_view(session_view, cell_ids)

            # Write to cache file
            cache_file = get_session_cache_file(
                Path(notebook_path).resolve()
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(session_data, indent=2))

            status = "with errors" if did_error else "ok"
            print(f"  -> {cache_file} ({status})")
            succeeded += 1

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()