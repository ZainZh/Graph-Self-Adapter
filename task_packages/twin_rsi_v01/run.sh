#!/usr/bin/env bash
set -euo pipefail
export_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python3}" "$export_root/tools/task_graph/twin_export_packages.py" launch --export-root "$export_root" "$@"
