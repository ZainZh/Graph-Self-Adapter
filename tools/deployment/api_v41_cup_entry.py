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
    stages[-1]['next']=None;g['stages']=stages;return g
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
        import cup_clear_goalset_v41 as runtime
        return runtime
    return old_load(path,name)
parent.entry.load=load
HERE=Path(__file__).resolve().parent
import all_code_stage_runtime
all_code_stage_runtime.CODE_STAGE_IDS=frozenset(all_code_stage_runtime.CODE_STAGE_IDS)|{'c02.07b.clear_retreat'}
parent.entry.SOURCES+=(Path(__file__),HERE/'cup_probe_lift_v37_runtime.py',HERE/'cup_visible_surfaces_v39.py',HERE/'cup_clear_retreat_v40_runtime.py',HERE/'cup_clear_goalset_v41.py',HERE/'cup_visible_plane_v36.py',HERE/'sim_preview_v33.py',HERE/'task_stream_view_v35.py')
def main(argv=None):return parent.main(argv)

from all_code_stage_route import validate_route as original_code_route
from prove.xtrainer.io import SafetyError

def extended_code_route(graph):
    extra=[s for s in graph['stages'] if s['stage_id']=='c02.07b.clear_retreat']
    if len(extra)!=1 or extra[0]['arm']!='right' or extra[0]['phase']!='Retreat' or extra[0]['controller']!='code_empty_gripper_clearance':raise SafetyError('exact extra clearance Code node required')
    report=original_code_route({**graph,'stages':[s for s in graph['stages'] if s['stage_id']!='c02.07b.clear_retreat']})
    report.update(graph_nodes=26,code_trajectory_stage_ids=report['code_trajectory_stage_ids']+['c02.07b.clear_retreat'],graph_or_weight_bytes_modified=True,weights_modified=False)
    return report
