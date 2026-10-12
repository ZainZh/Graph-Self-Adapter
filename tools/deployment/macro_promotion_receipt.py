#!/usr/bin/env python3
"""Root-workstation-only attestation after sealed strict full-task success.

No simulator or hardware is constructed. No receipt can be made from failure.
The laptop verifies package identity, never claims to rerun simulator evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path(__file__).resolve().parents[2]
RUNTIME_PATHS=('tools/deployment/handle_macro_entry.py','tools/deployment/handle_macro_runtime.py','tools/deployment/handle_macro_route.py',
 'tools/deployment/handle_full_runtime.py','tools/deployment/shared_cup_runtime.py','tools/deployment/object_centered_runtime.py',
 'prove/xtrainer/runtime.py','prove/xtrainer/agent.py','prove/xtrainer/perception.py','tools/deployment/handle_local_candidate_v4.py',
 'tools/deployment/handle_local_driver_v4r4.py','tools/deployment/handle_full_route_plan.py','tools/deployment/rgbd_capture_retry.py')
DIRECT_RUNTIME_PATHS=('tools/deployment/direct_white_entry.py','tools/deployment/direct_white_runtime.py','tools/deployment/direct_white_route.py','tools/deployment/direct_white_geometry.py')+RUNTIME_PATHS

ADDITIONAL_PATHS=('prove/xtrainer/io.py','prove/xtrainer/portable.py',
 'tools/deployment/shared_cup_entry.py','tools/deployment/operator_launcher.py','tools/deployment/real_feedback_stream.py',
 'tools/deployment/macro_promotion_check.py','tools/deployment/promoted_real_entry_check.py','tools/deployment/promoted_real_entry.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_success(freeze_path,result_path):
    freeze=json.loads(Path(freeze_path).read_text());result=json.loads(Path(result_path).read_text())
    if freeze.get('status')!='result_free_frozen' or result.get('strict_success') is not True or result.get('graph_completed') is not True:
        raise ValueError('sealed strict full-task simulator success required')
    if result.get('mask_source')!='actual_SAM2' or result.get('simulation_physics_continues_during_Root') is not True:raise ValueError('continuous actual-SAM simulator evidence absent')
    if result.get('shared_runtime_sha256')!=freeze.get('runtime_sha256'):raise ValueError('simulation runtime identity mismatch')
    bindings={str(Path(row['path']).resolve()):row['sha256'] for row in freeze['bindings']}
    if len(bindings)!=len(freeze['bindings']) or any(digest(path)!=sha for path,sha in bindings.items()):raise ValueError('original simulation freeze/source changed')
    if len(freeze.get('jobs',[]))!=1 or freeze['jobs'][0].get('mode')!='sim':raise ValueError('exact original simulator job required')
    output=Path(freeze['jobs'][0]['output']).resolve()
    if Path(result_path).resolve()!=output/'RESULT.json':raise ValueError('result outside frozen simulator job')
    seal=output/'PREDICTIONS_SEALED.json'
    sealed=json.loads(seal.read_text())
    if sealed.get('sealed_before_private_evaluator') is not True:raise ValueError('prediction-before-evaluation seal absent')
    config_path=next(Path(p) for p in bindings if Path(p).name=='CONFIG.json' and Path(p).parent.name!='cup')
    cfg=json.loads(config_path.read_text());graph_path=Path(cfg.get('direct_route_graph') or cfg['macro_route_graph'])
    if bindings.get(str(graph_path.resolve()))!=digest(graph_path):raise ValueError('qualified graph absent from original freeze')
    graph=json.loads(graph_path.read_text());expected=[stage['stage_id'] for stage in graph['stages']]
    if result.get('graph_stages')!=len(expected) or [row['stage_id'] for row in result['stage_trace'] if row.get('completed') is True]!=expected:raise ValueError('all declared graph nodes in original order required')
    return freeze,result,bindings,seal


def build(freeze_path,result_path,destination,root_identity):
    freeze,result,bindings,seal=verified_success(freeze_path,result_path)
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('fresh deployment package required')
    variant=result.get('execution_variant')
    runtime_paths=DIRECT_RUNTIME_PATHS if variant=='red_legacy_white_direct_box_v15' else RUNTIME_PATHS if variant=='handle_macro_continuous_transfer_v1' else None
    if runtime_paths is None:raise ValueError('unsupported qualified runtime variant')
    # Current canonical code must be precisely the code frozen during success.
    for rel in runtime_paths:
        if bindings.get(str((REPO/rel).resolve()))!=digest(REPO/rel):raise ValueError('qualified runtime file absent from original freeze: '+rel)
    runtime=hashlib.sha256(''.join(digest(REPO/p) for p in runtime_paths).encode()).hexdigest()
    if runtime!=freeze['runtime_sha256']:raise ValueError('canonical runtime hash differs from successful study')
    config_path=next(Path(p) for p in bindings if Path(p).name=='CONFIG.json' and Path(p).parent.name!='cup')
    cfg=json.loads(config_path.read_text());original_package=Path(cfg['portable_package'])
    for rel in ADDITIONAL_PATHS:
        candidates=(str((REPO/rel).resolve()),str((original_package/rel).resolve()))
        if rel not in ('tools/deployment/macro_promotion_check.py','tools/deployment/promoted_real_entry_check.py','tools/deployment/promoted_real_entry.py') and not any(bindings.get(path)==digest(REPO/rel) for path in candidates):raise ValueError('qualified support source mismatch: '+rel)
    destination.mkdir(parents=True);manifest=[]
    for rel in runtime_paths+ADDITIONAL_PATHS:
        source=REPO/rel;target=destination/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest.append({'relative_path':str(target.relative_to(destination)),'sha256':digest(target),'role':'canonical-runtime-source' if rel in runtime_paths else 'support-or-check-layer'})
    for rel in ('prove/__init__.py','prove/xtrainer/__init__.py'):
        source=original_package/rel;target=destination/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest.append({'relative_path':str(target.relative_to(destination)),'sha256':digest(target),'role':'verified portable-package namespace scaffold; foundation namespace unused by macro'})
    shutil.copytree(original_package,destination/'package')
    for target in sorted((destination/'package').rglob('*')):
        if target.is_file():manifest.append({'relative_path':str(target.relative_to(destination)),'sha256':digest(target),'role':'unchanged-versioned-portable-package'})
    data={'TASK_GRAPH.json':Path(cfg.get('direct_route_graph') or cfg['macro_route_graph']),'ROBOT_FLAT_PAD_GEOMETRY.json':Path(cfg['runtime']['robot_pad_geometry_file'])}
    prior=cfg['runtime'].get('static_reference_points_file')
    if prior:data['SAM_STATIC_REFERENCE.json']=Path(prior)
    for name,source in data.items():
        target=destination/'data'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest.append({'relative_path':str(target.relative_to(destination)),'sha256':digest(target),'role':'qualified-static-data'})
    launcher=destination/'run.sh';shutil.copyfile(REPO/'tools/deployment/promoted_run.sh',launcher)
    manifest.append({'relative_path':'run.sh','sha256':digest(launcher),'role':'independent real launcher; same canonical controller'})
    checkpoint=Path(cfg['runtime']['vision']['sam_checkpoint']);utility=Path(cfg['runtime']['vision']['sam_utility'])
    external={key:{'sha256':digest(cfg['runtime'][key])} for key in ('sdk_file','left_calibration','right_calibration')}
    external.update(sam_checkpoint={'sha256':digest(checkpoint),'config':cfg['runtime']['vision']['sam_config']},sam_utility={'sha256':digest(utility)})
    receipt={'schema':'graph-self-adapter.macro-promotion-receipt.v1','status':'Root-attested-engineering-simulation-pass','root_attestor_identity':root_identity,
        'task':'put_mug_on_box','execution_variant':variant,'graph_nodes':result['graph_stages'],'shared_runtime_sha256':runtime,'runtime_relative_paths':['source/'+p for p in runtime_paths],
        'files':manifest,'origin_evidence':{'full_freeze_sha256':digest(freeze_path),'full_result_sha256':digest(result_path),'prediction_seal_sha256':digest(seal)},
        'external_resources':external,
        'portable_package_manifest_sha256':digest(original_package/'PACKAGE_MANIFEST.json'),'scene_assets_packaged':False,
        'laptop_verification_contract':'verify these relative source/graph/model/reference hashes and Root attestation; no simulator reevaluation claim',
        'physical_hardware_qualification':False,'hardware_execution_authorized_by_receipt':False,'manual_current_Root_stage_judgement_required':True}
    (destination/'PROMOTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    local=destination/'local';local.mkdir()
    (local/'ROOT_RECEIPT_SHA256.txt').write_text(digest(destination/'PROMOTION_RECEIPT.json')+'\n')
    real=cfg['runtime'].copy();real['vision']=dict(real['vision'])
    for key in ('sdk_file','left_calibration','right_calibration'):
        real[key]=real[key].replace('/home/zheng','/home/clover')
    real['sdk_file']='/home/clover/github/xtrainer_clover/dobot_control/robots/dobot_api.py'
    for key in ('sam_root','sam_checkpoint','sam_utility'):real['vision'][key]=real['vision'][key].replace('/home/zheng','/home/clover')
    real.update(robot_TCP_offset_m=.197,hardware_safety_reviewed=False,initial_gripper_state_confirmed=False)
    real['vision'].update(role_boxes={},prompt_frame_sha256=None)
    (local/'CONFIG_REAL.json').write_text(json.dumps(real,indent=2)+'\n')
    (local/'SCOPE_REAL.json').write_text(json.dumps({'schema':'graph-self-adapter.promoted-real-scope.v1','task':'put_mug_on_box','read_only_shadow_authorized':True,'hardware_motion_commands_authorized':True,'authorization_source':'existing explicit user task to execute same validated controller on connected real robot; execute mode is operator motion declaration','physical_review_or_initial_jaw_measurement_inferred':False},indent=2)+'\n')
    shutil.copyfile(REPO/'tools/deployment/promoted_run.sh',destination/'run.sh')
    (destination/'EXPECTED_START_NOTE.txt').write_text('Known initial gripper values are expected commands from the declared prior, not measurements. Current robot q need not match demo q0; actual fresh stationary state is checked. No automatic Home/Enable/tool change/gripper scan. Execute mode declares motion; initial command values are not a measured jaw state and no review flags are manufactured. Current Root must inspect live Grounding and supported-boundary replies.\n')
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--root-identity',required=True);a=p.parse_args(argv)
    receipt=build(a.freeze,a.result,a.output,a.root_identity);print(json.dumps({'output':str(a.output),'runtime_sha256':receipt['shared_runtime_sha256'],'hardware_connections':0},indent=2))

if __name__=='__main__':main()
