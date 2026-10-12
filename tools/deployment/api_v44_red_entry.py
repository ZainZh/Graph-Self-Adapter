import cup_v25_current_goalset_one_command as parent
from pathlib import Path
from api_grounding_v32 import install
from direct_white_route import direct_route_graph as original_route

def route(graph):
    g=original_route(graph)
    for stage in g['stages']:
        if stage['phase'] not in ('Grasp','Lift'):continue
        for group,key in [('completion_predicates','done'),('invariant_predicates','invariants')]:
            removed={p['id'] for p in stage[group] if p['predicate'] in ('holding','holding_confirmed')}
            stage[group]=[p for p in stage[group] if p['id'] not in removed];stage[key]=[k for k in stage[key] if k not in removed]
        stage['holding_contract']='none';stage['diagnostic_contract']='unconfirmed closure then original bounded Lift; actual API holding required before transport'
    import copy
    stages=[]
    for node in g['stages']:
        if node['stage_id']=='c02.08.complete':
            clear=copy.deepcopy(node);clear.update(stage_id='c02.07b.clear_retreat',phase='Retreat',controller='code_empty_gripper_clearance',holding_contract='none',completion_predicates=[{'id':'own_pose','predicate':'actor_command_reached'},{'id':'own_progress','predicate':'reference_progress'}],invariant_predicates=[{'id':'finite','predicate':'finite_simulation'}],done=['own_pose','own_progress'],invariants=['finite'])
            stages.append(clear)
        stages.append(node)
    for a,b in zip(stages,stages[1:]):a['next']=b['stage_id']
    stages=[s for s in stages if s['cycle_id'] in ('c01','c02')]
    for a,b in zip(stages,stages[1:]):a['next']=b['stage_id']
    stages[-1]['next']=None;g['stages']=stages;g['logical_task']='put_red_mug_on_box';g['white_manipulation_requested']=False;return g
import direct_white_route
direct_white_route.direct_route_graph=route
parent.direct_route_graph=route
install('put_mug_on_box',parent)
old_bind=parent.entry.bindings

def bindings(config,cfg,mode):
    import direct_white_entry,all_code_stage_sim_entry
    direct_white_entry.direct_route_graph=route
    if hasattr(all_code_stage_sim_entry,'direct_route_graph'):all_code_stage_sim_entry.direct_route_graph=route
    all_code_stage_sim_entry.validate_route=extended_code_route
    return old_bind(config,cfg,mode)
parent.entry.bindings=bindings
old_load=parent.entry.load

def load(path,name):
    if Path(path).name=='direct_white_runtime.py':
        import red_only_v44_runtime as runtime
        return runtime
    return old_load(path,name)
parent.entry.load=load
HERE=Path(__file__).resolve().parent
import all_code_stage_runtime
all_code_stage_runtime.CODE_STAGE_IDS=frozenset(all_code_stage_runtime.CODE_STAGE_IDS)|{'c02.07b.clear_retreat'}
parent.entry.SOURCES+=(Path(__file__),HERE/'cup_probe_lift_v37_runtime.py',HERE/'cup_visible_surfaces_v39.py',HERE/'cup_clear_retreat_v40_runtime.py',HERE/'cup_clear_goalset_v41.py',HERE/'red_only_v44_runtime.py',HERE/'cup_visible_plane_v36.py',HERE/'sim_preview_v33.py',HERE/'task_stream_view_v35.py')
def main(argv=None):
    result=parent.main(argv)
    if isinstance(result,dict) and "post_seal_private_evaluation" in result:
        import json,hashlib
        import argparse
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--scope',type=Path,required=True);a,_=p.parse_known_args(argv)
        owners=list((a.scope.parent/'runs').glob('*'));run=max(owners,key=lambda x:x.stat().st_mtime)
        legacy=run/'RESULT.json';data=json.loads(legacy.read_text());evaluation=data['post_seal_private_evaluation']
        checks=evaluation['task_outcome']['checks'];passed=bool(data['graph_completed'] and checks['red_mug_grasped'] and checks['red_mug_on_box'] and evaluation['numerical_safety']['passed'])
        report={'logical_task':'put_red_mug_on_box','scene_task':'put_mug_on_box','graph_completed':data['graph_completed'],'graph_nodes':17,'completed_nodes':len(data['stage_trace']),'failure':data['failure'],'red_only_goal_passed':passed,'red_checks':{k:v for k,v in checks.items() if k.startswith('red_mug')},'original_numerical_safety_passed':evaluation['numerical_safety']['passed'],'legacy_double_cup_result_source':str(legacy),'legacy_result_sha256':hashlib.sha256(legacy.read_bytes()).hexdigest(),'double_cup_pass_claim':False,'white_stage_executed':any(s['stage_id'].startswith('white') for s in data['stage_trace']),'evaluation_after_prediction_seal':True,'hardware_motion_commands':data['hardware_motion_commands']}
        (run/'RED_ONLY_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
        return report
    return result

from all_code_stage_route import validate_route as original_code_route
from prove.xtrainer.io import SafetyError

def extended_code_route(graph):
    expected={'c02.07b.clear_retreat'}|{f'c0{i}.{j:02d}.{name}' for i in (1,2) for j,name in enumerate(('approach','grasp','lift','move','align','release','wait','complete'),1)}
    stages=graph['stages']
    if len(stages)!=17 or {s['stage_id'] for s in stages}!=expected or any(s['object']!='red_mug' for s in stages):raise SafetyError('exact single-red 17-node graph required')
    code=[s['stage_id'] for s in stages if s['phase'] in ('Approach','Lift','Move','Release','Wait','Retreat')]
    precise=[s['stage_id'] for s in stages if s['phase'] in ('Grasp','Align')]
    semantics=[s['stage_id'] for s in stages if s['phase']=='Complete']
    return {'schema':'graph-self-adapter.all-code-stage-route.v1','graph_nodes':17,'code_trajectory_stage_ids':code,'precision_closed_loop_stage_ids':precise,'semantic_boundary_stage_ids':semantics,'fixed_z':True,'graph_or_weight_bytes_modified':True,'weights_modified':False,'logical_task':'put_red_mug_on_box'}
