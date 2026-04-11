#!/usr/bin/env bash
set -euo pipefail

BASE='Read README.md and TODO.html. Find the first incomplete TODO subtask. Use conda env "midi_mover". No Python file may exceed 600 lines.'

mkdir -p .cline-plans

for i in $(seq 1 30); do
  echo "=== PLAN $i / 21 ==="
  cline -y -p --timeout 1800 \
    "$BASE Create only a concise implementation plan. Do not modify files." \
    > ".cline-plans/plan-$i.md"

  echo "=== ACT $i / 30 ==="
  cline -y -a --timeout 3600 \
    "$BASE Follow this approved plan exactly and implement it fully:

$(cat ".cline-plans/plan-$i.md")"
done