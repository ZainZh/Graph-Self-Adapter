#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
CUP_RUNTIME_PYTHON=${CUP_PYTHON:-/home/zheng/miniforge3/envs/geometry/bin/python}
exec "$CUP_RUNTIME_PYTHON" "$TASK_ROOT/tools/deployment/macro_one_command.py" "$@"
