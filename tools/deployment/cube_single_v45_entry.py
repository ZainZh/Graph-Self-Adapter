"""Scoped single-cube variant; same public sensor contract and declared uncertain control attachment."""
import copy,json
from pathlib import Path
import ordinary_v25_cube_state_guard_entry as parent
from ordinary_v25_cube_stationary_judge_route import observer_route as old_route
import api_grounding_v34 as grounding

def route(graph):
    raw=copy.deepcopy(graph);raw['stages']=[s for s in raw['stages'] if s['cycle_id']=='c01'];g=old_route(raw)
    for s in g['stages']:
        if s['phase'] in ('Grasp','Lift'):s.update(holding_contract='none',done=['own_pose','own_progress'],invariants=['finite'])
    g['semantic_provider']='openrouter/openai/gpt-6-astra';g['logical_task']='collect_single_cube';return g
parent.ordinary_route=route;parent.entry.base.ordinary_route=route;parent.entry.base.role_contract=parent.entry.roles
oldload=parent.entry.loader

def load(path,name):
    if Path(path).name=='direct_white_runtime.py':
        import cube_single_v45_runtime as runtime
        return runtime
    return oldload(path,name)
parent.entry.loader=load
# Actual API initial localisation, never hardcoded boxes or evaluator masks.
grounding.ROLES['collect_cubes']={'yellow_cube':'whole yellow cube','bowl':'whole receiving bowl','table_patch':'small bare unobstructed tabletop patch near yellow cube, avoid robot/objects'}
grounding.install('collect_cubes',parent)
original_main=parent.entry.backend.main

def backend_main(argv=None):
    from cube_api_judge_v45 import APIStageJudge
    parent.entry.backend.RealRootJudge=lambda *a,**kw:APIStageJudge(*a,**kw,maximum_calls=3)
    return original_main(argv)
parent.entry.backend.main=backend_main
HERE=Path(__file__).resolve().parent
parent.entry.base.SOURCES+=(Path(__file__),HERE/'cube_single_v45_runtime.py',HERE/'cube_api_judge_v45.py',HERE/'task_cube_v45.py',HERE/'sim_preview_v33.py',HERE/'task_stream_view_v35.py')
def main(argv=None):return parent.main(argv)
