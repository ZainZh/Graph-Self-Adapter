#!/usr/bin/env python3
"""Prepare minimal task package now, attest only after Root-bound strict modern pass."""
import argparse,json,shutil
from pathlib import Path
from prove.xtrainer.agent import digest
from prove.xtrainer.io import SafetyError
from prove.xtrainer.multitask_bundle import load_bundle
from multitask_ordinary_route import ordinary_route
import multitask_ordinary_entry_v21 as motion
from ordinary_real_portable_v21 import SCHEMA

TASKS=('collect_apple','collect_cubes','put_knife_board_new','sort_two_spoon')
EXTRA=('ordinary_real_portable_v21.py','ordinary_real_entry_v21.py')


def build(task,config,destination):
    if task not in TASKS:raise SafetyError('ordinary worker owns these four tasks only')
    b=load_bundle(task);cfg=json.loads(Path(config).read_text());root=Path(destination).resolve()
    if cfg.get('task')!=task or root.exists():raise SafetyError('exact task config/fresh package required')
    root.mkdir(parents=True);files=[];repo=motion.REPO
    def copy(path,relative):
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);files.append({'path':relative,'sha256':digest(target)})
    for path in motion.SOURCES:copy(path,'source/'+str(path.relative_to(repo)))
    for name in EXTRA:copy(motion.HERE/name,'source/tools/deployment/'+name)
    for relative in ('prove/__init__.py','prove/xtrainer/__init__.py'):
        if (repo/relative).exists():copy(repo/relative,'source/'+relative)
        else:
            target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('');files.append({'path':'source/'+relative,'sha256':digest(target)})
    assets={}
    for key,name in [('graph','TASK_GRAPH.json'),('parent_graph','PARENT_TASK_GRAPH.json'),('model','local_skill.npz'),('references','visible_centroids.json'),('candidate_freeze','CANDIDATE_FREEZE.json'),('skill_source','frozen_local_skill.py'),('motor_prior','motor_targets.npz')]:
        relative='data/'+name;assets[key]=relative
        if key=='graph':
            target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(ordinary_route(b['graph']),indent=2)+'\n');files.append({'path':relative,'sha256':digest(target)})
        else:copy(b['paths']['graph'] if key=='parent_graph' else b['paths'][key],relative)
    launcher='''#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"; if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODE" == real ]]; then MODE=execute; fi
exec "${XTRAINER_PYTHON:-/home/clover/miniconda3/envs/geometry/bin/python}" "$PACKAGE_ROOT/source/tools/deployment/ordinary_real_entry_v21.py" --task '''+task+''' "$MODE" --package "$PACKAGE_ROOT" "$@"
'''
    (root/'run.sh').write_text(launcher);files.append({'path':'run.sh','sha256':digest(root/'run.sh')})
    runtime=cfg['runtime'];external={key:digest(runtime[key]) for key in ('sdk_file','left_calibration','right_calibration')};external.update(sam_utility=digest(runtime['vision']['sam_utility']),sam_checkpoint=digest(runtime['vision']['sam_checkpoint']))
    manifest={'schema':SCHEMA,'task':task,'fixed_z':True,'files':files,'assets':assets,'runtime_relative_paths':['source/'+str(p.relative_to(repo)) for p in motion.SOURCES],'runtime_sha256':motion.runtime_hash(),'external_resource_sha256':external,'graph_nodes':len(ordinary_route(b['graph'])['stages']),'simulation_qualification':'pending until same-runtime full modern success; legacy GT checks do not qualify','standard_scene_packaged':False,'key_material_packaged':False}
    (root/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');local=root/'local';local.mkdir()
    real=json.loads(json.dumps(runtime));real['mode']='execute';real['known_initial_gripper_commands']=b['initial_g'].tolist()
    for key in ('sdk_file','left_calibration','right_calibration'):real[key]=real[key].replace('/home/zheng','/home/clover')
    real['sdk_file']='/home/clover/github/xtrainer_clover/dobot_control/robots/dobot_api.py'
    for key in ('sam_root','sam_checkpoint','sam_utility'):real['vision'][key]=real['vision'][key].replace('/home/zheng','/home/clover')
    real['stage_judge']['key_file']='/home/clover/.config/graph-self-adapter/openrouter.key';real['stage_judge'].pop('protocol_file',None)
    (local/'CONFIG_REAL.json').write_text(json.dumps(real,indent=2)+'\n')
    (local/'SCOPE_REAL.json').write_text(json.dumps({'task':task,'read_only_shadow_authorized':True,'hardware_motion_commands_authorized':True,'authorization':'existing user task; explicit execute is operator motion intent, no fabricated review/measurement'},indent=2)+'\n')
    (root/'README.txt').write_text('Default check: files only, no key/device. shadow/real require same-runtime full-task strict simulator receipt; pending packages refuse before connecting. Current Root supplies fresh initial role boxes; supported Grasp and exact Complete use Gemini. No Enable/Home/tool or automatic gripper initialization. Source/model/reference bound per task; key external at /home/clover/.config/graph-self-adapter/openrouter.key. No real grasp/communication/depth qualification inferred. Retain failures in RESULT, COMMAND_INTENT (ACK0 may include motion), STAGE_TRACE and REAL_AGENT_HANDOFFS/*/API_CALL/CONSUMED.\n')
    return manifest


def promote(root,freeze_path,result_path,root_identity):
    root=Path(root);manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text());freeze=json.loads(Path(freeze_path).read_text());result=json.loads(Path(result_path).read_text())
    if freeze.get('status')!='result_free_frozen' or not result.get('strict_success') or not result.get('graph_completed') or result.get('mask_source')!='actual_SAM2':raise SafetyError('sealed strict modern full-task simulation required')
    bound={str(Path(r['path']).resolve()):r['sha256'] for r in freeze['bindings']}
    if any(digest(path)!=sha for path,sha in bound.items()):raise SafetyError('original full-task source binding changed')
    if freeze.get('jobs')!=[{'task':manifest['task'],'mode':'sim','output':str(Path(result_path).resolve().parent)}]:raise SafetyError('exact task simulator job binding required')
    seal=Path(result_path).parent/'PREDICTIONS_SEALED.json'
    if not json.loads(seal.read_text()).get('sealed_before_private_evaluator'):raise SafetyError('postseal evaluator ordering absent')
    if result.get('task')!=manifest['task'] or result.get('shared_runtime_sha256')!=manifest['runtime_sha256']:raise SafetyError('task/runtime differs from modern pass')
    cfgpath=next(Path(path) for path in bound if Path(path).name=='CONFIG.json' and Path(path).parent.name!='cup');cfg=json.loads(cfgpath.read_text())
    expected_graph=json.loads(Path(cfg['ordinary_route_graph']).read_text());packaged_graph=json.loads((root/manifest['assets']['graph']).read_text())
    if expected_graph!=packaged_graph or [r['stage_id'] for r in result['stage_trace'] if r.get('completed')]!=[r['stage_id'] for r in expected_graph['stages']]:raise SafetyError('all exact task graph nodes must complete')
    for relative in manifest['runtime_relative_paths']:
        original=motion.REPO/Path(relative).relative_to('source')
        if bound.get(str(original))!=digest(root/relative):raise SafetyError('portable runtime not the frozen successful source')
    receipt={'task':manifest['task'],'strict_success':True,'runtime_sha256':manifest['runtime_sha256'],'package_manifest_sha256':digest(root/'PACKAGE_MANIFEST.json'),'root_attestor_identity':root_identity,'origin_evidence':{'freeze':digest(freeze_path),'result':digest(result_path),'seal':digest(seal)},'physical_hardware_qualified':False}
    (root/'PROMOTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');(root/'local/ROOT_RECEIPT_SHA256.txt').write_text(digest(root/'PROMOTION_RECEIPT.json')+'\n');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.task,a.config,a.output),indent=2))
