#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$package_dir/../.." && pwd)"
exec "${CUP_PYTHON:-/home/zheng/miniforge3/envs/geometry/bin/python}" "$repo_dir/tools/deployment/shared_cup_entry.py" "$@"
