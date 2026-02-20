#!/usr/bin/env bash
# Validate that every marimo notebook has a session JSON file and that
# session JSONs don't contain known error patterns.
#
# Usage:
#   scripts/validate-sessions.sh
#
# Exits with code 1 if any check fails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOKS_DIR="${REPO_ROOT}/notebooks"

# cd to repo root so all paths are relative
cd "$REPO_ROOT"

# When running in GitHub Actions, emit ::error annotations.
# Locally, just print plain error messages.
error() {
  local file="$1"
  local msg="$2"
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::error file=${file}::${msg}"
  else
    echo "ERROR: ${msg}"
  fi
}

exit_code=0

# ---------------------------------------------------------------------------
# 1. Check every notebook has a session JSON
# ---------------------------------------------------------------------------
echo "Checking that every notebook has a session JSON..."

while IFS= read -r nb; do
  # Skip non-marimo files
  if ! grep -q "marimo.App" "$nb"; then
    continue
  fi

  dir=$(dirname "$nb")
  base=$(basename "$nb")
  expected="${dir}/__marimo__/session/${base}.json"

  if [ ! -f "$expected" ]; then
    error "$nb" "Missing session JSON: ${expected}"
    exit_code=1
  fi
done < <(find notebooks -name '*.py' \
  -not -path '*/inputs/*' \
  -not -name '__init__.py')

if [ "$exit_code" -eq 0 ]; then
  echo "  All notebooks have session JSON files."
fi

# ---------------------------------------------------------------------------
# 2. Check session JSONs for error patterns
# ---------------------------------------------------------------------------
echo "Checking session JSONs for errors..."

error_patterns="ModuleNotFoundError"

found_errors=0
while IFS= read -r json_file; do
  # Derive the notebook path from the session JSON path
  # e.g. notebooks/foo/__marimo__/session/bar.py.json -> notebooks/foo/bar.py
  notebook=$(echo "$json_file" | sed 's|/__marimo__/session/|/|; s|\.json$||')

  for pattern in $error_patterns; do
    if grep -q "$pattern" "$json_file"; then
      error "$notebook" "Notebook '${notebook}' has '${pattern}' in its session JSON (${json_file})"
      exit_code=1
      found_errors=1
    fi
  done
done < <(find notebooks -name '*.json' -path '*/__marimo__/session/*')

if [ "$found_errors" -eq 0 ]; then
  echo "  All session JSON files are clean."
fi

exit $exit_code
