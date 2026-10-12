#!/usr/bin/env bash
set -euo pipefail
DEPLOYMENT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROMOTED_PYTHON=${CUP_PYTHON:-/home/clover/miniconda3/envs/geometry/bin/python}
mode=${1:-check}
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" == real ]]; then mode=execute; fi
exec "$PROMOTED_PYTHON" "$DEPLOYMENT_ROOT/source/tools/deployment/promoted_real_entry.py" "$mode" --package "$DEPLOYMENT_ROOT" "$@"
