"""Independent preparation only; never imported by active local03.

This does not execute a robot, construct a simulator, or grant full-task scope.
"""
import copy
import ast
import hashlib
import json
from pathlib import Path


WHITE_CYCLES=('c03','c04')


def graph_copy_with_handle_route(original):
    routed=copy.deepcopy(original);changed=[]
    for stage in routed['stages']:
        if stage['cycle_id'] in WHITE_CYCLES and stage['object']=='white_mug' and stage['phase'] in ('Approach','Grasp','Lift'):
            stage['engineering_route']='observed_handle_tube_Code_flat_v4'
            if stage['phase']=='Grasp':stage['controller']='code_handle_tube'
            if stage['phase']=='Lift':stage['controller']='code_observed_envelope_clearance'
            changed.append(stage['stage_id'])
    if len(changed)!=6:raise ValueError('expected exactly white c03/c04 Approach/Grasp/Lift')
    routed['engineering_route_contract']={'version':'white-handle-Code-flat-v4','original_stage_order_preserved':True,
        'changed_stage_ids':changed,'neural_claim':'white Grasp is explicit functional-part Code, not learned handle policy',
        'first_white_grounding':'scene-initial actual SAM functional patch and explicit Root seed',
        'second_white_grounding':'fresh current RGB Grounding and actual SAM reinitialization required; no prior world point reuse',
        'arm_contract':'stage actor arm; robot-own pad FK and native/shared transform; passive actual hold',
        'handoff':'actual successful Grasp attachment; observed-sphere clearance Lift; common accepted-Lift attachment refresh then original Move/Align/Release/Wait',
        'full_execution_gate':'new scope/freeze and successful local v3 physical qualification with exact source hashes'}
    return routed


def checked_local_qualification(result_path,freeze_path,*,candidate=3,code_source=None,driver_source=None,pad_artifact=None):
    result=json.loads(Path(result_path).read_text());freeze=json.loads(Path(freeze_path).read_text())
    sources=(Path(code_source) if code_source else Path(__file__).resolve().with_name(f'handle_local_candidate_v{candidate}.py'),Path(driver_source) if driver_source else Path(__file__).resolve().with_name(f'handle_local_driver_v{candidate}.py'))
    declared=[ast.literal_eval(node.value) for node in ast.parse(sources[1].read_text()).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VARIANT' for t in node.targets)]
    if len(declared)!=1 or not isinstance(declared[0],str):raise ValueError('qualified driver variant declaration absent')
    expected_variant=declared[0]
    if result.get('variant')!=expected_variant or result.get('candidate')!=candidate or result.get('local_qualified') is not True:
        raise ValueError('candidate3 local qualification has not passed')
    qualification=result.get('post_seal_local_qualification',{})
    if qualification.get('passed') is not True or qualification.get('private_post_seal_only') is not True or qualification.get('physical_window_sec',0)<59.8:
        raise ValueError('continuous local physical qualification absent')
    bindings={str(Path(row['path']).resolve()):row['sha256'] for row in freeze['bindings']}
    job=freeze.get('jobs',[])
    if len(job)!=1 or job[0].get('candidate')!=candidate or job[0].get('kind')!='handle_local':raise ValueError('exact qualified candidate job absent')
    for source in sources:
        if bindings.get(str(source))!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('qualified candidate3 source changed')
    if pad_artifact is not None:
        artifact=json.loads(Path(pad_artifact).read_text())
        bound=artifact.get('source_binding')
        if not bound or bound.get('sha256')!=hashlib.sha256(sources[0].read_bytes()).hexdigest():raise ValueError('qualified pad artifact extractor binding mismatch')
    return {'qualified_result_path':str(Path(result_path).resolve()),'qualified_result_sha256':hashlib.sha256(Path(result_path).read_bytes()).hexdigest(),
            'qualification_freeze_path':str(Path(freeze_path).resolve()),'qualification_freeze_sha256':hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest()}
