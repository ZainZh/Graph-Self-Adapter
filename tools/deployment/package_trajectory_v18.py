#!/usr/bin/env python3
"""File-only V18 deployment preparation from sealed strict success."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from macro_promotion_receipt import verified_success,DIRECT_RUNTIME_PATHS,ADDITIONAL_PATHS,digest,REPO
import trajectory_entry_v18 as entry

EXTRA=('tools/deployment/trajectory_promotion_check.py','tools/deployment/promoted_trajectory_real_entry.py')

def build(freeze_path,result_path,destination,root_identity):
    freeze,result,bound,seal=verified_success(freeze_path,result_path)
    if result['execution_variant']!='white_stage_trajectory_v18_lower_contact' or freeze['runtime_sha256']!=entry.runtime_hash():raise ValueError('exact successful V18 runtime required')
    root=Path(destination).resolve()
    if root.exists():raise FileExistsError('fresh package destination required')
    config_path=next(Path(p) for p in bound if Path(p).name=='CONFIG.json' and Path(p).parent.name!='cup')
    cfg=json.loads(config_path.read_text());original=Path(cfg['portable_package'])
    extension=tuple(str(p.relative_to(REPO)) for p in entry.SOURCES)
    canonical=tuple(dict.fromkeys(DIRECT_RUNTIME_PATHS+extension))
    for rel in canonical:
        if bound.get(str(REPO/rel))!=digest(REPO/rel):raise ValueError('source absent from successful freeze: '+rel)
    root.mkdir(parents=True);manifest=[]
    def copy(source,relative,role):
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest.append({'relative_path':relative,'sha256':digest(target),'role':role})
    support=tuple(p for p in ADDITIONAL_PATHS if 'promotion' not in p and 'promoted_real_entry' not in p)+EXTRA
    for rel in tuple(dict.fromkeys(canonical+support)):
        if rel not in EXTRA and rel not in canonical and bound.get(str(REPO/rel))!=digest(REPO/rel) and bound.get(str(original/rel))!=digest(REPO/rel):raise ValueError('support source differs from successful freeze: '+rel)
        copy(REPO/rel,'source/'+rel,'frozen-runtime' if rel in canonical else 'support-or-portable-entry')
    for rel in ('prove/__init__.py','prove/xtrainer/__init__.py'):copy(original/rel,'source/'+rel,'portable namespace')
    # The existing 20-file package owns graph/model/reference hashes; no standard scene.
    shutil.copytree(original,root/'package',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for target in sorted((root/'package').rglob('*')):
        if target.is_file():manifest.append({'relative_path':str(target.relative_to(root)),'sha256':digest(target),'role':'unchanged-portable-model-reference'})
    for name,path in {'TASK_GRAPH.json':cfg['direct_route_graph'],'ROBOT_FLAT_PAD_GEOMETRY.json':cfg['runtime']['robot_pad_geometry_file'],'SAM_STATIC_REFERENCE.json':cfg['runtime']['static_reference_points_file']}.items():copy(path,'data/'+name,'frozen-observation-prior-or-route')
    launcher='''#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODE" == real ]]; then MODE=execute; fi
exec "${XTRAINER_PYTHON:-/home/clover/miniconda3/envs/geometry/bin/python}" "$PACKAGE_ROOT/source/tools/deployment/promoted_trajectory_real_entry.py" "$MODE" --package "$PACKAGE_ROOT" "$@"
'''
    (root/'run.sh').write_text(launcher);manifest.append({'relative_path':'run.sh','sha256':digest(root/'run.sh'),'role':'independent V18 real launcher; default file-only'})
    runtime=cfg['runtime'];external={key:{'sha256':digest(runtime[key])} for key in ('sdk_file','left_calibration','right_calibration')}
    external.update(sam_checkpoint={'sha256':digest(runtime['vision']['sam_checkpoint']),'config':runtime['vision']['sam_config']},sam_utility={'sha256':digest(runtime['vision']['sam_utility'])})
    receipt={'schema':'graph-self-adapter.trajectory-promotion-receipt.v1','status':'Root-attested-engineering-simulation-pass','root_attestor_identity':root_identity,'task':'put_mug_on_box','execution_variant':result['execution_variant'],'graph_nodes':result['graph_stages'],'shared_runtime_sha256':entry.runtime_hash(),'base_runtime_relative_paths':['source/'+p for p in DIRECT_RUNTIME_PATHS],'extension_runtime_relative_paths':['source/'+p for p in extension],'runtime_relative_paths':['source/'+p for p in canonical],'files':manifest,'external_resources':external,'origin_evidence':{'full_freeze_sha256':digest(freeze_path),'full_result_sha256':digest(result_path),'prediction_seal_sha256':digest(seal)},'scene_assets_packaged':False,'physical_hardware_qualification':False,'hardware_execution_authorized_by_receipt':False,'manual_current_Root_stage_judgement_required':True}
    (root/'PROMOTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    local=root/'local';local.mkdir();(local/'ROOT_RECEIPT_SHA256.txt').write_text(digest(root/'PROMOTION_RECEIPT.json')+'\n')
    real=json.loads(json.dumps(runtime));real['robot_TCP_offset_m']=.197
    for key in ('sdk_file','left_calibration','right_calibration'):real[key]=real[key].replace('/home/zheng','/home/clover')
    real['sdk_file']='/home/clover/github/xtrainer_clover/dobot_control/robots/dobot_api.py'
    for key in ('sam_root','sam_checkpoint','sam_utility'):real['vision'][key]=real['vision'][key].replace('/home/zheng','/home/clover')
    real['vision'].update(role_boxes={},prompt_frame_sha256=None)
    (local/'CONFIG_REAL.json').write_text(json.dumps(real,indent=2)+'\n')
    scope={'schema':'graph-self-adapter.promoted-real-scope.v1','task':'put_mug_on_box','read_only_shadow_authorized':True,'hardware_motion_commands_authorized':True,'authorization_source':'existing session user whole-task real execution authorization; explicit real/execute is operator intent, no action performed by package preparation','physical_review_or_initial_jaw_measurement_inferred':False}
    (local/'SCOPE_REAL.json').write_text(json.dumps(scope,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root-identity',required=True);a=p.parse_args()
    receipt=build(a.freeze,a.result,a.output,a.root_identity);print(json.dumps({'output':str(a.output),'runtime_sha256':receipt['shared_runtime_sha256'],'hardware_connections':0},indent=2))
