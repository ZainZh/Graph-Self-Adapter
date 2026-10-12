"""Explicit engineering completion contract; original28 internal action order stays."""
import copy
from handle_full_route_plan import graph_copy_with_handle_route


def macro_route_graph(original):
    graph=graph_copy_with_handle_route(original)
    for stage in graph['stages']:
        if stage['phase'] in ('Lift','Move','Align'):
            removed={row['id'] for row in stage['completion_predicates'] if row['predicate']=='stage_goal'}
            stage['completion_predicates']=[row for row in stage['completion_predicates'] if row['id'] not in removed]
            stage['done']=[key for key in stage['done'] if key not in removed]
            stage['engineering_completion']='own reference/Cartesian pose progress plus tentative macro grasp state; no visual object-goal claim'
            stage['holding_contract']='tentative_macro_grasp'
        if stage['phase']=='Grasp':stage['Agent_boundary']='Root visual holding at supported pick boundary'
        if stage['phase']=='Release':
            removed={row['id'] for row in stage['invariant_predicates'] if row['predicate'] in ('holding','holding_confirmed','stage_goal')}
            stage['invariant_predicates']=[row for row in stage['invariant_predicates'] if row['id'] not in removed]
            stage['invariants']=[key for key in stage['invariants'] if key not in removed]
            stage['holding_contract']='none'
            stage['Agent_boundary']='Root actual visual release/placement after commanded opening at declared support boundary'
    graph['macro_contract']={'variant':'continuous_transfer_macro_v1','cycles':4,'internal_nodes':28,
        'Root_boundaries':['Grasp','Release'],'unsupported_wait_for_Agent':False,
        'internal_semantics':'lifted/align_ready/support not asserted; actual object completion only released/placement Root boundary and original postseal private goal',
        'free_hold_60sec_gate':False,'motion_safety':'actual own-q freshness/limits, finite, gripper/visible anomaly checks; standardforce unchanged',
        'unpromoted_simulator_only':True}
    return graph
