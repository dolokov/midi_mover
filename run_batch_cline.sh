#!/usr/bin/env bash
set -euo pipefail

MAX_RUNS="${1:-21}"
LOG_DIR=".cline-logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"

mkdir -p "$LOG_DIR"

if ! command -v cline >/dev/null 2>&1; then
  echo "Error: 'cline' is not installed or not on PATH."
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: 'jq' is required."
  echo "Install with: sudo apt install jq"
  exit 1
fi

for i in $(seq 1 "$MAX_RUNS"); do
  echo "=== RUN $i / $MAX_RUNS ==="

  LOG_FILE="$LOG_DIR/run-$i.jsonl"

  PROMPT=$(cat <<'EOF'
Read README.md and TODO.html.
Find the first incomplete TODO subtask.

If there are no incomplete TODO subtasks left:
- do not modify any files
- your final response must be exactly: NO_TODOS_LEFT

Otherwise:
- implement exactly that one first incomplete TODO subtask completely
- always use conda env "midi_mover" for Python commands and tests
- do not leave any Python file longer than 600 lines
- run the relevant checks/tests
- update TODO.html to mark that one subtask done
- when finished, print a short summary of files changed and tests run
EOF
)

  cline -y -a --json --timeout "$TIMEOUT_SECS" "$PROMPT" | tee "$LOG_FILE"

  FINAL_TEXT="$(
    jq -r '
      select(.type=="say" and .say=="completion_result")
      | .text
    ' "$LOG_FILE" | tail -n 1
  )"

  if [[ "$FINAL_TEXT" == "NO_TODOS_LEFT" ]]; then
    echo
    echo "No TODOs left. Stopping after run $i."
    exit 0
  fi

  echo
done

echo "Reached max runs ($MAX_RUNS)."