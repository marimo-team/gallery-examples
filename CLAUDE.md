# Gallery Examples

This repo is a collection of [marimo](https://marimo.io) notebooks organized by category under `notebooks/`.

## Adding a new notebook

1. **Create the notebook file** in the appropriate category folder (e.g. `notebooks/external/`, `notebooks/math/`, etc.):
   - If you're given a github URL, use wget on the appropriate address to fetch it locally
   - Use PEP 723 inline script metadata at the top for dependencies
   - Follow the marimo `App` structure with `@app.cell` decorators
   - Pin dependency versions where possible
   - Run `uvx marimo check <notebook>` to check for mistakes

2. **Generate the session metadata** (pre-computed outputs for the gallery):
   ```bash
   uv run scripts/create-sessions.py notebooks/<category>/<notebook>.py
   ```
   This creates a JSON file at `notebooks/<category>/__marimo__/session/<notebook>.py.json`.

3. **Update the README** by adding an entry to the appropriate table section with:
   - Notebook name
   - Short description
   - molab link badge: `[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/<category>/<notebook>.py)`

## Useful scripts

- `uv run scripts/create-sessions.py <notebook paths...>` — generate session JSON for specific notebooks
- `bash scripts/create-sessions-changed.sh` — regenerate sessions for git-changed notebooks only
- `bash scripts/create-sessions-changed.sh -a` — regenerate all sessions
- `uv run scripts/validate-sessions.py` — validate that all session files are fresh and error-free
