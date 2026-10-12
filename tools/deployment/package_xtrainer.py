#!/usr/bin/env python3
"""Create a relocatable cup-only package from verified frozen A2 bytes."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from prove.xtrainer.runtime import load_bundle
from prove.xtrainer.agent import digest,write
from prove.xtrainer.portable import SCHEMA,verify_package

CORE=('io.py','perception.py','agent.py','runtime.py','mock.py','portable.py')
README='''# Portable Xtrainer cup package

This package preserves the actual frozen A2 cup graph, 29D NumPy model, visible references, frozen TwinLocalSkill and motor prior. It requires no MuJoCo scene, robot meshes or workstation task-plan path. Every asset and implementation file is hash-bound by PACKAGE_MANIFEST.json. Fixed-z, native rad/m units, limited DLS, 0.1 hardware reference scale, original initial-pose checks, real feedback freshness, bounded Root review and fail-stop remain in the existing Executor.

Default `./run.sh` or `./run.sh --check` only checks files; it connects no devices. Set XTRAINER_PYTHON to an existing environment containing numpy/scipy/PIL/yaml/zmq/torch/pyzed. Existing robot project and SAM installation are external; no SDK construction, enable/home, camera or gripper action occurs during check.

```bash
XTRAINER_PYTHON=/path/to/env/bin/python ROBOT_PROJECT=/path/to/RoboTwin_bin_pick SAM_ROOT=/path/to/SAM2_streaming SAM_CHECKPOINT=/path/to/sam2.1_hiera_large.pt ./run.sh --check
./run.sh --mock --output /tmp/new-cup-mock
./run.sh --prepare-freeze --freeze-mode shadow --scope /path/to/reviewed_scope.json --output /path/to/future-shadow > prepared-freeze.json
./run.sh --shadow --scope /path/to/reviewed_scope.json --execution-freeze /path/to/result-free-frozen.json --output /path/to/new-shadow
./run.sh --execute --config /path/to/reviewed_config.json --scope /path/to/reviewed_scope.json --execution-freeze /path/to/result-free-frozen.json --output /path/to/new-task
```

Use identical external-path overrides for prepare/check/shadow/execute. Preparing a freeze only binds actual package/config/calibration/SDK/SAM bytes and the future exact job. It does not authorize connections or motion, mark safety reviewed, confirm the initial gripper state or freeze itself. Root/operator must review scope and exact bindings and set status=result_free_frozen explicitly. Connection modes refuse absent authorization. Execute additionally refuses unreviewed physical safety or unconfirmed initial gripper state. Both flags remain false in CONFIG_PROPOSED.json. Do not set them automatically; there is no homing/teleport workaround.

SAM first-frame boxes must label the actual current exported RGB in the same process; initialization output is discarded and a fresh tracked image follows. The default prompt files are under /tmp. Real actual aperture stays unknown because service grip feedback is a placeholder. Actual q/qdot/TCP come from independent controller feedback; mode/User0/Tool1, joint step/velocity/tracking/workspace and pre-send freshness are checked. Shadow compares native actual TCP and controller FK and sends no motion. Collision planning and hardware task performance remain unqualified.

**The stage judge is the current chat Root, not an automatic model service.** Keep this Codex/chat session connected to the laptop's result files, or manually share each REAL_AGENT_HANDOFFS request's actual RGB panel/frame files with Root. Root must actually inspect the current/past RGB and produce ROOT_REPLY.json plus independent ROOT_APPROVAL.json using the real-stage schemas from prove/xtrainer/agent.py; do not use canned/model-service/mock replies for hardware. While Root reviews, no new command is sent, a read-only feedback heartbeat checks safety/drift, and a reply triggers fresh RGB-D/feedback validation. UNKNOWN never authorizes true facts. Time/request budgets stop if review stalls. A single shell command does not provide unattended semantic inference.

The local robot ZMQ service must already be running on 127.0.0.1:6001. This package never starts it or constructs DobotRobot. Native command packs Lq6+Lg+Rq6+Rg, with 0 closed/1 open gripper commands. ResetRobot is the failure-stop path, never enable/home. Mock is an explicitly synthetic integration fixture, not hardware or Root reasoning evidence.
'''
LAUNCHER='''#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xtrainer_python="${XTRAINER_PYTHON:-python3}"
extra=()
if [[ -n "${ROBOT_PROJECT:-}" ]]; then extra+=(--robot-project "$ROBOT_PROJECT"); fi
if [[ -n "${SAM_ROOT:-}" ]]; then extra+=(--sam-root "$SAM_ROOT"); fi
if [[ -n "${SAM_CHECKPOINT:-}" ]]; then extra+=(--sam-checkpoint "$SAM_CHECKPOINT"); fi
exec "$xtrainer_python" "$package_dir/tools/task_graph/xtrainer_portable.py" --package "$package_dir" "${extra[@]}" "$@"
'''


def package(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('fresh portable package directory required')
    bundle=load_bundle('put_mug_on_box');output.mkdir(parents=True)
    assets={'graph/task_graph.json':'graph','models/local_skill.npz':'model','refs/visible_centroids.json':'references',
            'source/frozen_twin_local_skill.py':'skill_source','motor_targets.npz':'motor_prior','CANDIDATE_FREEZE.json':'candidate_freeze'}
    for relative,key in assets.items():
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(bundle['paths'][key],destination)
    for name in CORE:
        destination=output/'prove/xtrainer'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/'prove/xtrainer'/name,destination)
    destination=output/'tools/task_graph/xtrainer_portable.py';destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(REPO/'tools/task_graph/xtrainer_portable.py',destination)
    for relative in ('prove/__init__.py','prove/xtrainer/__init__.py','tools/__init__.py','tools/task_graph/__init__.py'):
        (output/relative).write_text('')
    shutil.copyfile(REPO/'task_packages/xtrainer_v01/CONFIG_PROPOSED.json',output/'CONFIG_PROPOSED.json')
    (output/'README.md').write_text(README);(output/'run.sh').write_text(LAUNCHER);(output/'run.sh').chmod(0o755)
    files=[{'path':str(p.relative_to(output)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()]
    manifest={'schema':SCHEMA,'version':'xtrainer-laptop-v01','task':'put_mug_on_box','fixed_z':True,
              'original_motor_sha256':digest(bundle['paths']['motor_prior']),'files':files,
              'packager_source_sha256':digest(Path(__file__)),'standard_scene_assets_included':False,
              'SAM_weights_included':False,'hardware_connections_made':0,'hardware_motion_commands':0}
    write(output/'PACKAGE_MANIFEST.json',manifest);verify_package(output)
    archive=output.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz') as tar:tar.add(output,arcname=output.name)
    return {'package':str(output),'archive':str(archive),'archive_sha256':digest(archive),
            'manifest_sha256':digest(output/'PACKAGE_MANIFEST.json'),'files':len(files),'hardware_connections':0}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);result=package(a.output);print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
