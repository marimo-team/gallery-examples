#!/usr/bin/env bash
# Run create-sessions.py in parallel for marimo notebooks.
#
# Usage:
#   bash scripts/create-sessions-changed.sh              # only git-changed notebooks
#   bash scripts/create-sessions-changed.sh -a            # all notebooks
#   bash scripts/create-sessions-changed.sh -a -j 8       # all notebooks, 8 parallel jobs

set -eo pipefail

MAX_JOBS=4
ALL=false

while getopts "aj:" opt; do
    case $opt in
        a) ALL=true ;;
        j) MAX_JOBS=$OPTARG ;;
        *) echo "Usage: $0 [-a] [-j max_parallel_jobs]" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT="$REPO_ROOT/scripts/create-sessions.py"

changed_files=()
if [ "$ALL" = true ]; then
    # Find all marimo notebooks (py files containing "import marimo"), excluding scripts/
    while IFS= read -r f; do
        changed_files+=("$f")
    done < <(grep -rl "import marimo" --include='*.py' "$REPO_ROOT/notebooks" | sed "s|^$REPO_ROOT/||" | sort)
else
    # Collect notebooks with uncommitted changes (staged + unstaged), deduplicated
    while IFS= read -r f; do
        changed_files+=("$f")
    done < <({ git diff --name-only -- '*.py'; git diff --cached --name-only -- '*.py'; } | sort -u)
fi

if [ ${#changed_files[@]} -eq 0 ]; then
    if [ "$ALL" = true ]; then
        echo "No marimo notebooks found."
    else
        echo "No uncommitted .py files found."
    fi
    exit 0
fi

label=$( [ "$ALL" = true ] && echo "total" || echo "changed" )
echo "Found ${#changed_files[@]} $label notebook(s), running up to $MAX_JOBS in parallel"
echo ""

LOGDIR=$(mktemp -d)
trap 'rm -rf "$LOGDIR"' EXIT

succeeded=()
failed=()
running=0
pids=()
files=()

reap_finished() {
    for i in "${!pids[@]}"; do
        if ! kill -0 "${pids[$i]}" 2>/dev/null; then
            wait "${pids[$i]}" && rc=0 || rc=$?
            if [ $rc -eq 0 ]; then
                echo "  OK:   ${files[$i]}"
                succeeded+=("${files[$i]}")
            else
                echo "  FAIL: ${files[$i]} (exit $rc)"
                failed+=("${files[$i]}")
            fi
            unset 'pids[i]' 'files[i]'
            ((running--))
            # Re-index sparse arrays
            pids=("${pids[@]}")
            files=("${files[@]}")
            return 0
        fi
    done
    return 1
}

for file in "${changed_files[@]}"; do
    full_path="$REPO_ROOT/$file"

    # Wait if we've hit the parallelism limit
    while [ $running -ge $MAX_JOBS ]; do
        reap_finished || sleep 0.1
    done

    echo "Starting: $file"
    logfile="$LOGDIR/$(basename "$file").log"
    uv run "$SCRIPT" "$full_path" > "$logfile" 2>&1 &
    pids+=($!)
    files+=("$file")
    ((running++))
done

# Wait for remaining jobs
while [ $running -gt 0 ]; do
    reap_finished || sleep 0.1
done

echo ""
echo "================================="
echo "  ${#succeeded[@]} succeeded, ${#failed[@]} failed out of ${#changed_files[@]}"
echo "================================="

if [ ${#failed[@]} -gt 0 ]; then
    echo ""
    echo "Failed notebooks:"
    for f in "${failed[@]}"; do
        echo "  - $f (log: $LOGDIR/$(basename "$f").log)"
    done
    # Don't clean up logs on failure
    trap - EXIT
    exit 1
fi
