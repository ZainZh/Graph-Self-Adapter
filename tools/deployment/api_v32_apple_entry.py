import ordinary_v25_apple_pad_entry as parent
from api_grounding_v32 import install
from pathlib import Path
from multitask_ordinary_route import ordinary_route as old_route
from multitask_ordinary_route import role_contract

def route(graph):
    g=old_route(graph)
    for stage in g['stages']:
        if stage['phase'] in ('Grasp','Lift'):
            stage['holding_contract']='none';stage['done']=['own_pose','own_progress'];stage['invariants']=['finite']
            stage['completion_source']='own pose/progress; no holding assertion; Lift requires API confirmation before Move'
        if stage['phase']=='Lift':stage['diagnostic_action']='bounded original reference Lift; holding unknown until post-Lift API judge'
    g['revision']='v30 retained causal closure-lift boundary';return g
parent.ordinary_route=route;parent.entry.base.ordinary_route=route
install('collect_apple',parent)
old_load=parent.entry.loader

def load(path,name):
    if Path(path).name=='direct_white_runtime.py':
        import apple_probe_lift_v30_runtime as runtime
        return runtime
    return old_load(path,name)
parent.entry.loader=load;parent.entry.base.SOURCES+=(Path(__file__),Path(__file__).with_name('apple_probe_lift_v30_runtime.py'))
def main(argv=None):return parent.main(argv)
