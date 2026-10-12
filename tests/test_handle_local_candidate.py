import importlib.util
from pathlib import Path
import numpy as np
import pytest
from prove.xtrainer.io import SafetyError
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_candidate',ROOT/'tools/deployment/handle_local_candidate.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def test_functional_patch_uses_actual_mask_depth_and_rejects_missing():
    depth=np.full((10,10),.5);mask=np.zeros((10,10),bool);mask[2:8,2:8]=True;K=np.array([[100,0,5],[0,100,5],[0,0,1.]])
    points=candidate.observed_patch_cloud(depth,mask,K,np.eye(4),[2,2,5,8]);assert len(points)==18
    with pytest.raises(SafetyError,match='insufficient'):candidate.observed_patch_cloud(depth,mask*False,K,np.eye(4),[2,2,5,8])


def test_tube_goal_matches_pad_midpoint_and_translates_with_observed_part():
    rng=np.random.default_rng(2);points=np.c_[rng.normal(0,.0005,80),rng.normal(0,.0005,80),np.linspace(.04,.08,80)]+[.1,.2,0]
    rel=candidate.tube_relation(points,[.1,-1,.5]);pads={'closing_axis_TCP_local':np.array([1.,0,0]),'pad_midpoint_TCP_local_m':np.array([.01,0,-.015])}
    p,R=candidate.candidate_goal(rel,pads);assert np.allclose(p+R@pads['pad_midpoint_TCP_local_m'],rel['center']);assert np.allclose(R.T@R,np.eye(3));assert np.linalg.det(R)>0.999
    shift=np.array([.03,0,0]);rel2=candidate.tube_relation(points+shift,np.array([.1,-1,.5])+shift);p2,R2=candidate.candidate_goal(rel2,pads);assert np.allclose(p2-p,shift) and np.allclose(R,R2)


def test_lift_depends_on_observed_height_and_table_without_constant_script():
    table=np.c_[np.linspace(0,.1,40),np.zeros(40),np.zeros(40)];cloud=np.c_[np.zeros(40),np.zeros(40),np.linspace(0,.08,40)];pose=(np.zeros(3),np.eye(3))
    goal,a=candidate.lift_goal_from_visible_height(pose,cloud,table);goal2,b=candidate.lift_goal_from_visible_height(pose,cloud*1.5,table)
    assert goal[0][2]>0 and goal2[0][2]>goal[0][2] and a['requested_rise_m']!=.02


def test_exact_candidate_freeze_rejects_wrong_job_without_devices(tmp_path):
    import json
    hs=importlib.util.spec_from_file_location('_handle_driver',ROOT/'tools/deployment/handle_local_driver.py');driver=importlib.util.module_from_spec(hs);hs.loader.exec_module(driver)
    scope=tmp_path/'scope.json';scope.write_text(json.dumps({'execution_authorized':True,'mode':'sim','hardware_motion_jobs':0,'maximum_local_candidate_episodes':3}))
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps({'schema':'graph-self-adapter.handle-local-freeze.v1','status':'result_free_frozen','jobs':[],'scope_sha256':driver.digest(scope)}))
    with pytest.raises(SafetyError,match='exact local'):driver.guard(scope,freeze,{},tmp_path/'output')


def test_local_entry_precision_rejects_three_mm_even_when_old_controller_would_pass():
    hs=importlib.util.spec_from_file_location('_handle_precision_driver',ROOT/'tools/deployment/handle_local_driver.py');driver=importlib.util.module_from_spec(hs);hs.loader.exec_module(driver)
    executor=driver.LocalExecutor.__new__(driver.LocalExecutor);executor.entry_position_tolerance_m=.0007
    goal=(np.zeros(3),np.eye(3));assert not executor._reached(goal,(np.array([.002,0,0]),np.eye(3)))
    assert executor._reached(goal,(np.array([.0005,0,0]),np.eye(3)))


def test_actual_local_completion_sequence_includes_approach(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import threading
    hs=importlib.util.spec_from_file_location('_handle_sequence_driver',ROOT/'tools/deployment/handle_local_driver.py');driver=importlib.util.module_from_spec(hs);hs.loader.exec_module(driver)
    ex=driver.LocalExecutor.__new__(driver.LocalExecutor);clock=[0.];ex.clock=lambda:clock[0];ex.sleep=lambda dt:clock.__setitem__(0,clock[0]+dt)
    sample=SimpleNamespace(q=np.zeros(6),tcp=np.array([0.,0.,.1]),rotation=np.eye(3));obs=SimpleNamespace(roles={'white_mug':np.zeros(3)})
    cloud=np.c_[np.zeros(40),np.zeros(40),np.linspace(0,.08,40)];table=cloud.copy();table[:,2]=0.
    ex.vision=SimpleNamespace(config={'role_boxes':{'white_handle':[],'table_patch':[]}},T=np.eye(4),cloud=lambda role:table if role=='table_patch' else cloud)
    ex.command_g=np.ones(2);ex.b={'initial_g':np.ones(2)};ex.cfg={'safety':{'max_tracking_rad':.05}};ex.safety=SimpleNamespace(last_target=np.zeros((2,6)))
    ex.robot=SimpleNamespace(fk_lock=threading.Lock(),fk=lambda *a:None,world=SimpleNamespace(model=None),scratch=None);ex.stops=[];ex.trace=[];ex.output=tmp_path;ex._hardware_attempts=0;ex._hardware_acknowledged=0;ex.judge=SimpleNamespace(records=[])
    ex.start=lambda:None;ex.node=lambda phase:None;ex.capture=lambda:([sample,sample],obs);ex.reach=lambda *a:None;ex.dispatch=lambda *a:None;ex.confirm=lambda *a:None
    pads={'inner_pad_points_TCP_local_m':np.zeros((2,3)),'pad_vertices_TCP_local_m':np.zeros((2,3)),'pad_midpoint_TCP_local_m':np.zeros(3),'closing_axis_TCP_local':np.array([1.,0,0])}
    monkeypatch.setattr(driver,'pad_relation_from_robot_mesh',lambda *a:pads)
    monkeypatch.setattr(driver,'tube_relation',lambda *a:{'radial_extent_m':.004,'span_m':.02,'tangent_down':np.array([0.,0,-1.])})
    monkeypatch.setattr(driver,'candidate_goal',lambda *a:(np.array([0.,0,.05]),np.eye(3)))
    result=ex.local_run();assert result['completed_local_nodes']==['Approach','Grasp','Lift','Hold'] and result['local_completed'] is True
    assert result['graph_completed'] is False and result['continuous_hold_observed_wall_sec']>=60.


def test_postseal_qualification_rejects_unheld_table_cup_even_after_root_completion():
    from types import SimpleNamespace
    import mujoco
    hs=importlib.util.spec_from_file_location('_handle_qualification_driver',ROOT/'tools/deployment/handle_local_driver.py');driver=importlib.util.module_from_spec(hs);hs.loader.exec_module(driver)
    scene=Path('/home/zheng/github/Mujoco-RSI/runs/physics_v2/put_mug_on_box/scene.xml');model=mujoco.MjModel.from_xml_path(str(scene))
    rows=[{'physical_time_sec':float(t),'qpos':model.qpos0.copy(),'qvel':np.zeros(model.nv),'ctrl':np.zeros(model.nu)} for t in np.linspace(0,61,611)]
    robot=SimpleNamespace(mj=mujoco,world=SimpleNamespace(model=model,contact_evidence=lambda:{'roles':{'white_mug':{'held_by':{'left':False},'object_table_support_contact':True}}}))
    result=driver.qualify_after_seal(robot,rows,{'local_completed':True})
    assert result['passed'] is False and result['private_post_seal_only'] is True
