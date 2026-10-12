"""Explicit specialist routes; frozen phase labels/NN indexing are retained."""
import copy
from prove.xtrainer.io import SafetyError
from multitask_ordinary_route import ORDINARY,role_contract
SPECIAL=('Insert','Pull','Push','Hold')
CARRY=('Lift','Move','Align','Insert','Pull','Push','Hold')


def special_route(graph):
    stages=graph['stages'];phases={s['phase'] for s in stages}
    if not phases<=set(ORDINARY+SPECIAL):raise SafetyError('special phase handler absent')
    result=copy.deepcopy(graph);routed=[];declared_tools=[];held_task=graph['task'] in ('sweep_objects','twin_sweep_objects')
    for original in stages:
        stage=copy.deepcopy(original);phase=stage['phase']
        stage.update(completion_predicates=[],invariant_predicates=[],done=['own_pose','own_progress'],invariants=['finite'],completion_source='own pose/progress and tentative tool state; no hidden contact/goal')
        if phase in CARRY:stage['invariants'].append('tentative_holding')
        if phase=='Grasp':stage['done'].append('tentative_holding')
        if phase=='Release':stage['done'].append('opening_progress')
        if held_task and stage['arm']=='right':stage['invariants'].append('passive_tools_held')
        if phase=='Grasp':declared_tools.append({'arm':stage['arm'],'object':stage['object'],'actor':stage['actor']})
        routed.append(stage)
        terminal=phase=='Wait' or held_task and phase=='Hold'
        if terminal:
            complete=copy.deepcopy(stage);held=held_task and phase=='Hold'
            complete.update(stage_id=stage['cycle_id']+'.complete',phase='Complete',controller='exact_supported_or_held_goal_boundary',holding_contract='tools_held' if held else 'none',done=['holding','placement_complete'] if held else ['released','placement_complete'],invariants=['finite'],hold_at_end=held,declared_held_tools=copy.deepcopy(declared_tools) if held else [],semantic_facts=['holding','placement_complete'] if held else ['released','placement_complete'],completion_source='exact declared task relation; held terminal does not release')
            routed.append(complete)
    if not any(s['phase']=='Complete' for s in routed):raise SafetyError('special route missing explicit terminal boundary')
    for a,b in zip(routed,routed[1:]):a['next']=b['stage_id']
    routed[-1]['next']=None;result['stages']=routed;result['schema']='graph-self-adapter.multitask-special-route.v1';return result
