"""Explicit ordinary manipulation route; specialist phases need real handlers."""
import copy
from prove.xtrainer.io import SafetyError
ORDINARY=('Approach','Grasp','Lift','Move','Align','Release','Wait')
CARRY=('Lift','Move','Align')


def ordinary_route(graph):
    stages=graph['stages'];unsupported=sorted({s['phase'] for s in stages}-set(ORDINARY))
    if unsupported:raise SafetyError('task-specific handlers required for '+graph['task']+': '+','.join(unsupported))
    result=copy.deepcopy(graph);routed=[]
    for original in stages:
        stage=copy.deepcopy(original);phase=stage['phase']
        stage.update(completion_predicates=[],invariant_predicates=[],done=['own_pose','own_progress'],invariants=['finite'],completion_source='own Cartesian pose/progress; no synthetic lifted/align/goal truth')
        if phase in CARRY:stage['invariants'].append('tentative_holding')
        if phase=='Grasp':stage['done'].append('tentative_holding')
        if phase=='Release':stage['done'].append('opening_progress')
        routed.append(stage)
        if phase=='Wait':
            complete=copy.deepcopy(stage);complete.update(stage_id=stage['cycle_id']+'.complete',phase='Complete',controller='supported_semantic_goal_boundary',done=['released','placement_complete'],invariants=['finite'],reference_commands=0,semantic_facts=['released','placement_complete'],goal_relation=original['goal_relation'],completion_source='released and actual declared target relation after empty retreat')
            routed.append(complete)
    cycles={s['cycle_id'] for s in stages}
    for cycle in cycles:
        sequence=[s['phase'] for s in stages if s['cycle_id']==cycle]
        if sequence!=list(ORDINARY):raise SafetyError('ordinary route needs complete Approach→Grasp→Lift→Move→Align→Release→Wait cycle: '+cycle)
    for a,b in zip(routed,routed[1:]):a['next']=b['stage_id']
    routed[-1]['next']=None;result['stages']=routed;result['schema']='graph-self-adapter.multitask-ordinary-route.v1';result['parent_graph_task']=graph['task']
    return result


def role_contract(graph,refs):
    roles={s['object'] for s in graph['stages']}|{s['target'] for s in graph['stages']}
    synthetic=set(refs.get('synthetic_semantic_anchors',{}))|{'table_handover_region'}
    return {'SAM_roles':sorted((roles-synthetic)|{'table_patch'}),'fixed_declared_anchor_roles':sorted(roles&synthetic),'observed_table_role':'table_patch','grounding':'current Root boxes on retained fresh initial RGB; never evaluator masks'}
