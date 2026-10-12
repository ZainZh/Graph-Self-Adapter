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
    return g
import direct_white_route
direct_white_route.direct_route_graph=route
parent.direct_route_graph=route
install('put_mug_on_box',parent)
old_bind=parent.entry.bindings

def bindings(config,cfg,mode):
    import direct_white_entry,all_code_stage_sim_entry
    direct_white_entry.direct_route_graph=route
    if hasattr(all_code_stage_sim_entry,'direct_route_graph'):all_code_stage_sim_entry.direct_route_graph=route
    return old_bind(config,cfg,mode)
parent.entry.bindings=bindings
old_load=parent.entry.load

def load(path,name):
    if Path(path).name=='direct_white_runtime.py':
        import cup_probe_lift_v37_runtime as runtime
        return runtime
    return old_load(path,name)
parent.entry.load=load
HERE=Path(__file__).resolve().parent
parent.entry.SOURCES+=(Path(__file__),HERE/'cup_probe_lift_v37_runtime.py',HERE/'cup_visible_plane_v36.py',HERE/'sim_preview_v33.py',HERE/'task_stream_view_v35.py')
def main(argv=None):return parent.main(argv)
