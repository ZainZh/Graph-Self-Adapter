"""Generic bounded ordinary task variant, current RGB API grounding and explicit reduced goals."""
import copy,json
from pathlib import Path
import ordinary_v25_repair_entry as parent
from multitask_ordinary_route import ordinary_route as original_route
from api_grounding_v34 import install as install_api
import api_grounding_v34 as grounding
from prove.xtrainer.agent import digest
TASK=None;SINGLE=False

def route(graph):
    raw=copy.deepcopy(graph)
    if SINGLE:raw['stages']=[s for s in raw['stages'] if s['cycle_id']=='c01']
    g=original_route(raw)
    for s in g['stages']:
        if s['phase'] in ('Grasp','Lift'):s.update(holding_contract='none',done=['own_pose','own_progress'],invariants=['finite'],completion_source='unconfirmed closure then bounded original Lift; API verification before carry')
    g['variant']='single_first_object' if SINGLE else 'full';return g

def configure(task,single):
    global TASK,SINGLE
    TASK=task;SINGLE=single
    parent.ordinary_route=route;parent.entry.base.ordinary_route=route
    roles={
      'put_knife_board_new':{'red_knife':'whole red-handled knife including visible blade; exclude rack/other knife','board':'whole receiving wooden cutting board'},
      'sort_two_spoon':{'black_spoon':'whole black/dark spoon','plate':'whole receiving plate'}}
    if task=='sort_two_spoon' and not single:roles[task]['orange_spoon']='whole orange spoon'
    grounding.ROLES[task]=roles[task]
    install_api(task,parent)
    old=parent.entry.loader
    def load(path,name):
        if Path(path).name=='direct_white_runtime.py':
            import generic_probe_v46_runtime as runtime
            return runtime
        return old(path,name)
    parent.entry.loader=load
    parent.entry.base.SOURCES+=(Path(__file__),Path(__file__).with_name('generic_probe_v46_runtime.py'),Path(__file__).with_name('task_auto_v45.py'),Path(__file__).with_name('sim_preview_v33.py'),Path(__file__).with_name('task_stream_view_v33.py'),Path(__file__).with_name('task_stream_view_v35.py'))

def main(argv=None):
    result=parent.main(argv)
    if isinstance(result,dict) and 'post_seal_private_evaluation' in result:
        return result
    return result

_original_bindings=parent.entry.bindings
def bindings(path,cfg,mode):
    result=_original_bindings(path,cfg,mode)
    reference=Path(cfg['runtime']['paired_SAM_reference']);result[3][str(reference.resolve())]=digest(reference)
    return result
parent.entry.bindings=bindings
