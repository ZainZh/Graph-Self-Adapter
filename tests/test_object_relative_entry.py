import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=Path(__file__).resolve().parents[1]/'tools/deployment/object_relative_entry.py'
spec=importlib.util.spec_from_file_location('_object_entry_test',p);planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)


def fk(q,g,arm):
    assert arm=='left'
    return np.asarray(q)[0,:3].copy(),Rotation.from_rotvec(np.asarray(q)[0,3:]).as_matrix()


def plan(start,obj=None,**kw):
    reference=np.array([[.2,.1,.3,.1,-.2,.05],[.4,.4,.4,0,0,0]])
    return planner.plan_first_red_grasp_entry(actual_q=start,current_object_shared_m=[.35,.2,.1] if obj is None else obj,
        reference_q=reference,reference_object_shared_m=[.15,.05,.1],native_fk=fk,gripper_command=[1,1],
        T_left_from_right=np.eye(4),joint_min_rad=[-1]*6,joint_max_rad=[1]*6,
        workspace_min_m=[-1]*3,workspace_max_m=[1]*3,**kw)


def test_different_actual_initial_poses_reach_same_object_goal_without_demo_q0_gate():
    a=np.zeros((2,6));b=np.full((2,6),-.1)
    first,second=plan(a),plan(b)
    assert first['converged'] and second['converged']
    assert np.allclose(first['goal_TCP_shared_m'],[.4,.25,.3])
    assert first['goal_TCP_shared_m']==second['goal_TCP_shared_m']
    assert first['actual_start_q']!=second['actual_start_q'] and first['commands_sent']==0


def test_translated_object_translates_goal_and_reference_rotation_is_preserved():
    a=plan(np.zeros((2,6)));b=plan(np.zeros((2,6)),obj=[.38,.18,.11])
    assert b['converged'] and np.allclose(np.asarray(b['goal_TCP_shared_m'])-a['goal_TCP_shared_m'],[.03,-.02,.01])
    actual_rotation=np.asarray(b['poses'][-1]['rotation_shared']);reference=np.asarray(b['goal_rotation_shared'])
    assert np.linalg.norm(Rotation.from_matrix(reference@actual_rotation.T).as_rotvec())<.005


def test_candidate_steps_bounded_and_passive_arm_unchanged():
    start=np.zeros((2,6));result=plan(start);path=np.asarray(result['q_path'])
    assert np.max(abs(np.diff(path,axis=0)))<=.010000001
    assert np.all(path>=-1) and np.all(path<=1) and np.all(path[:,1]==start[1])
    assert result['freshness_and_collision_verified'] is False


def test_unreachable_joint_bound_retains_desired_goal_and_never_returns_unsafe_candidate():
    result=planner.plan_first_red_grasp_entry(actual_q=np.zeros((2,6)),current_object_shared_m=[.8,0,0],
        reference_q=np.zeros((2,6)),reference_object_shared_m=[0,0,0],native_fk=fk,gripper_command=[1,1],
        T_left_from_right=np.eye(4),joint_min_rad=[-.05]*6,joint_max_rad=[.05]*6)
    assert not result['converged'] and result['goal_TCP_shared_m']==[.8,0.,0.]
    assert np.max(abs(np.asarray(result['q_path'])))<=.05 and result['commands_sent']==0
