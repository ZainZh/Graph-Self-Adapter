#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${CUP_RUNTIME_PYTHON:-/home/zheng/miniforge3/envs/geometry/bin/python}" "$repo_root/tools/deployment/three_task_v26.py" --task put_mug_on_box "$@"
