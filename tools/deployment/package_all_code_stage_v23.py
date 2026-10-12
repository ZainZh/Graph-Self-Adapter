#!/usr/bin/env python3
"""Build the unvalidated V23 operator package from frozen V19 assets plus new source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "evaluations/gemini_graph_v19/deployment/cup_gemini_v19_r1"
NEW_SOURCE = (
    "tools/deployment/all_code_stage_route.py",
    "tools/deployment/all_code_stage_trajectory.py",
    "tools/deployment/all_code_stage_runtime.py",
    "tools/deployment/all_code_stage_sim_entry.py",
    "tools/deployment/all_code_stage_real_entry.py",
    "tools/deployment/all_code_stage_operator_entry.py",
    "tools/deployment/manual_sam_first_frame.py",
    "tools/deployment/wait_for_manual_sam.py",
    "tools/deployment/preview_saved_sam_masks.py",
    "tests/test_all_code_stage_runtime.py",
    "tests/test_manual_sam_first_frame.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(destination):
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError("fresh V23 package destination required")
    shutil.copytree(BASE, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs"))
    for relative in NEW_SOURCE:
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    fixture = root / "source/evaluations/gemini_graph_v19/control/20261007_111206_7f0994a9/TASK_GRAPH.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE / "data/TASK_GRAPH.json", fixture)
    shutil.copyfile(REPO / "tools/deployment/all_code_stage_sam_compat_bootstrap.py", root / "bootstrap.py")
    shutil.copyfile(REPO / "evaluations/cup_manual_stage_v23/DEPLOYMENT_AUTHORITY.json",
                    root / "data/DEPLOYMENT_AUTHORITY.json")
    config_path = root / "local/CONFIG_REAL.json"
    config = json.loads(config_path.read_text())
    config["vision"]["sam_checkpoint"] = "/home/clover/DataModel/sam2.1/sam2.1_hiera_small.pt"
    config["vision"].update(role_boxes={}, prompt_frame_sha256=None,
                            prompt_frame_output="RUN/INITIAL_RGB.png",
                            prompt_reply_path="RUN/INITIAL_BOXES.json")
    config["initial_gripper_state_confirmed"] = False
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    scope = {
        "schema": "graph-self-adapter.all-code-stage-operator-scope.v1",
        "task": "put_mug_on_box",
        "candidate_runtime_sha256": "f2efc4bd50ec717c77185654ae62f62830489fd027e7158ae781e000145991c3",
        "qualification": "unvalidated_operator_engineering_trial",
        "same_runtime_full_simulation_passed": False,
        "user_explicit_skip_simulation": True,
        "read_only_shadow_authorized": False,
        "hardware_motion_commands_authorized": True,
        "hardware_session_authorized": True,
        "explicit_user_direct_deploy_authorized": True,
        "authorization_source": "user: 别跑模拟了，你直接部署; existing current two-cup whole-task motion authorization retained",
        "deployment_authority_file": "data/DEPLOYMENT_AUTHORITY.json",
        "deployment_authority_sha256": digest(root / "data/DEPLOYMENT_AUTHORITY.json"),
        "physical_qualification_inferred": False,
    }
    (root / "local/SCOPE_REAL.json").write_text(json.dumps(scope, indent=2) + "\n")
    engineering = {
        "schema": "graph-self-adapter.all-code-stage-gui-repair-scope.v1",
        "deployment_revision": "cup_manual_stage_v23_r2",
        "parent_deployment": "cup_manual_stage_v23_r1",
        "parent_preserved": True,
        "change": "OpenCV 4.13 manual GUI window initialization only",
        "manual_window_id": "Manual-SAM-Grounding",
        "initial_render_before_callback": True,
        "control_sources_changed": False,
        "route": {"code_trajectory_nodes": 17, "precision_nodes": 5, "semantic_nodes": 3},
        "r1_failures": [
            {"run": "20261007_181343_execute", "hardware_commands": 0},
            {"run": "20261007_181436_execute", "hardware_commands": 0}
        ],
        "simulation_started": False,
        "robot_test_during_packaging": False
    }
    (root / "data/ENGINEERING_SCOPE.json").write_text(json.dumps(engineering, indent=2) + "\n")
    launcher = '''#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODE" == real ]]; then MODE=execute; fi
exec "$PACKAGE_ROOT/bootstrap.py" "$PACKAGE_ROOT/source/tools/deployment/all_code_stage_operator_entry.py" "$MODE" --package "$PACKAGE_ROOT" "$@"
'''
    (root / "run_v23.sh").write_text(launcher); (root / "run_v23.sh").chmod(0o755)
    (root / "bootstrap.py").chmod(0o755)
    (root / "README_V23.txt").write_text(
        "Default `bash run.sh check` verifies package/external files only: zero device/API.\n"
        "Operator hardware trial: `bash run.sh real --confirm-initial-grippers-open`.\n"
        "It captures current INITIAL_RGB, opens the manual five-box/one-seed GUI, then starts motion only after confirmation. Cancel/close/timeout is fail-closed.\n"
        "This candidate skipped same-runtime full simulation by explicit user instruction; it is an unvalidated operator engineering trial, not a scientific promotion or hardware qualification.\n"
    )
    files = []
    for folder in ("source", "package", "data", "local"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                files.append({"relative_path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size})
    for name in ("PROMOTION_RECEIPT.json", "README.txt", "run.sh", "README_V23.txt", "run_v23.sh"):
        path = root / name
        files.append({"relative_path": name, "sha256": digest(path), "bytes": path.stat().st_size})
    receipt = {
        "schema": "graph-self-adapter.all-code-stage-operator-package.v1",
        "deployment_revision": "cup_manual_stage_v23_r2",
        "parent_deployment_preserved": "cup_manual_stage_v23_r1",
        "task": "put_mug_on_box",
        "qualification": "unvalidated_operator_engineering_trial",
        "same_runtime_full_simulation_passed": False,
        "user_explicit_skip_simulation": True,
        "candidate_runtime_sha256": scope["candidate_runtime_sha256"],
        "base_assets_verified_by_unchanged_v19_receipt": True,
        "graph_nodes": 25,
        "code_trajectory_nodes": 17,
        "precision_closed_loop_nodes": 5,
        "semantic_boundaries": 3,
        "manual_initial_roles": ["white_mug", "red_mug", "box", "white_handle", "table_patch"],
        "manual_foreground_seed": "white_handle",
        "keys_packaged": False,
        "standard_simulator_assets_packaged": False,
        "files": files,
    }
    (root / "OPERATOR_TRIAL_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"output": str(root), "receipt_sha256": digest(root / "OPERATOR_TRIAL_RECEIPT.json"),
            "bootstrap_sha256": digest(root / "bootstrap.py"), "files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(build(args.output), indent=2))
