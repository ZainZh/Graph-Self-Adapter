#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON_BIN="${TRAJECTORY_PYTHON:-/home/zheng/miniforge3/envs/geometry/bin/python}"
exec "$PYTHON_BIN" "$ROOT_DIR/tools/deployment/trajectory_one_command.py" "$@"
