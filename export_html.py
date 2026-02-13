"""Export all marimo notebooks to HTML.

Usage:
    python export_html.py
"""

import subprocess
from pathlib import Path


def is_marimo_notebook(path: Path) -> bool:
    """Check if a Python file is a marimo notebook."""
    return "app = marimo.App" in path.read_text()


def main():
    notebooks_dir = Path("notebooks")
    notebooks = sorted(
        p for p in notebooks_dir.rglob("*.py") if is_marimo_notebook(p)
    )

    print(f"Found {len(notebooks)} marimo notebooks\n")

    for i, notebook in enumerate(notebooks, 1):
        output_dir = notebook.parent / "__marimo__"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / notebook.with_suffix(".html").name

        print(f"[{i}/{len(notebooks)}] {notebook}")
        result = subprocess.run(
            ["uvx", "marimo", "export", "html", str(notebook), "-o", str(output_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr.strip()}\n")
        else:
            print(f"  -> {output_file}\n")


if __name__ == "__main__":
    main()
