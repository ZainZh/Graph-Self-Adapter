import copy
import numpy as np
from prove.task_graph.p53_micro_insert import PublicPosePath,MicroTeacher,near_supervisor


def fixture():
    target=np.eye(4);target[2,3]=-.13
    relation=dict(final_caddy_book=target.tolist(),book_collision_center=[0,0,0],book_collision_halfsize=[.014,.05,.06],back_halfsize=[.028,.06,.06],mouth_clearance_margin=.001)
    book=np.eye(4);book[2,3]=-.162
    private=dict(world_caddy=np.eye(4),world_book=book,world_eef=book.copy(),rim_height=.1,finger_contacts=(True,True),official_success=False,wrong_target=False)
    return relation,private


def test_gate_excludes_below_rim_already_goal_and_one_finger():
    r,p=fixture();assert near_supervisor(r,p)['near_ready']
    p['world_book'][2,3]+=.005;assert not near_supervisor(r,p)['near_ready']
    r,p=fixture();p['official_success']=True;g=near_supervisor(r,p);assert g['code_completed_goal'] and not g['near_ready']
    r,p=fixture();p['finger_contacts']=(True,False);assert not near_supervisor(r,p)['near_ready']
    r,p=fixture();p['wrong_target']=True;assert not near_supervisor(r,p)['near_ready']


def test_public_pose_path_requires_geometric_arrival_not_elapsed_calls():
    refs=[dict(position=[0,0,0],quaternion_xyzw=[0,0,0,1]),dict(position=[.02,0,0],quaternion_xyzw=[0,0,0,1])]
    c=PublicPosePath(refs,.0008,.012)
    for _ in range(130):
        a,done,d=c.command([0,0,0],[0,0,0,1],0,1)
        assert not done and a[0]>.0 and a[6]==1
        assert np.linalg.norm(a[:3]*.05)<=.01000000001
    _,done,_=c.command([.02,0,0],[0,0,0,1],0,1);assert done


def test_micro_has_no_recovery_and_requires_goal_contact_hold():
    r,p=fixture();teacher=MicroTeacher(r)
    a,label=teacher.command(p);assert label['teacher_phase']=='micro_insert' and label['stop_reason'] is None
    p['world_book'][0,3]=.03
    _,label=teacher.command(p);assert label['stop_reason']=='micro_footprint_exit'
    r,p=fixture();teacher=MicroTeacher(r);p['official_success']=True
    for i in range(3):
        a,label=teacher.command(p);assert (label['stop_reason']=='qualified_goal_contact_hold')==(i==2)
        np.testing.assert_array_equal(a,[0,0,0,0,0,0,1])


def test_graph_allows_explicit_frozen_policy_route_but_rejects_unknown_backend():
    import json
    from pathlib import Path
    import pytest
    from tools.task_graph.collect_libero_p53 import validate_graph
    # Construct graph contract without reading a runtime environment or result.
    graph=dict(schema='prove.p53-short-taskgraph.v1',prefix_nodes=[dict(id=f's{i}') for i in range(1,7)],shared_code_budget=120,nodes=[
        dict(id='s7',skill='Align',backend='public_code_pose_path',next='s8',reference_start=0,reference_end=1,target_rule='raise_current_xy_orientation_to_public_max_z_then_reference41',termination='public_endpoint_reached'),
        dict(id='s8',skill='Approach',backend='public_code_pose_path',next='s9',reference_start=2,reference_end=2,target_rule='public_reference57',termination='public_endpoint_reached_and_near_ready'),
        dict(id='s9',skill='Insert',backend='frozen_stage_policy',next=None,max_actions=24,termination='original_goal_and_contact_hold3')])
    validate_graph(graph)
    graph['nodes'][2]['backend']='unbound'
    with pytest.raises(ValueError):validate_graph(graph)
