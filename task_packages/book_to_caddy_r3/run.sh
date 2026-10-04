#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
if [ "$#" -eq 0 ]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/task_bundle.py" run
else
  exec "$PYTHON_BIN" "$SCRIPT_DIR/task_bundle.py" "$@"
fi
