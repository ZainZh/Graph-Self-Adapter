import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from multitask_ordinary_runtime_v21 import RelativeLiftExecutor,relative_lift_goal
from prove.xtrainer.io import SafetyError


def test_translated_and_rotated_actual_grasp_starts_exactly_without_centroid_shift():
    actual=(np.array([.3,-.2,.1]),Rotation.from_euler('z',.5).as_matrix());reference=(np.array([.01,.02,.03]),np.eye(3))
    p,R=relative_lift_goal(actual,reference,reference)
    assert np.allclose(p,actual[0]) and np.allclose(R,actual[1])
    endpoint=(reference[0]+np.array([0.,0.,.06]),Rotation.from_euler('x',.04).as_matrix())
    p,R=relative_lift_goal(actual,reference,endpoint)
    assert np.allclose(p,actual[0]+[0.,0.,.06]) # yaw-only current pose cannot create lateral lift
    assert np.allclose(R,actual[1]@endpoint[1])


def test_first_lift_command_ignores_occluded_attachment_offset_and_preserves_closed_g(tmp_path):
    x=RelativeLiftExecutor.__new__(RelativeLiftExecutor);x.accepted_grasp={'cycle':'c1','g_command':0.};x.lift_progress={};x.relative_lift_records=[]
    actual=np.zeros((2,6));samples=[SimpleNamespace(q=q,tcp=np.array([.2,.1,.05]),rotation=np.eye(3)) for q in actual]
    x.accepted_grasp_pose={'c1':(samples[0].tcp.copy(),np.eye(3))};x.current_holding=True;x.holding=lambda *a:True;x.command_g=np.array([.1,.8]);x.T=np.eye(4)
    x.attachment={'a_actual':np.array([.03,-.03,.0])} # deliberately wrong; Lift must not use it
    x.b={'qcurve':lambda t:np.zeros((2,6))};x._shared_pose=lambda *a:(np.array([.01,.02,.03]),np.eye(3));goals=[]
    x._step_ik=lambda q,g,arm,goal:(goals.append(goal) or (q.copy(),None,np.zeros(6)))
    x.capture=lambda:(samples,SimpleNamespace(stamp=10.));x.cfg={'hardware_reference_speed_scale':1.,'safety':{'max_age_sec':.25}}
    x.clock=lambda:10.;x.sleep=lambda dt:None;x.safety=SimpleNamespace(validate=lambda *a:None,samples=lambda *a:None,last_target=None)
    issued=[];x.robot=SimpleNamespace(samples=lambda:samples,command=lambda q,g,flags:issued.append((q,g,flags)));x._hardware_attempts=0;x._hardware_acknowledged=0;x.command_index=0;x.output=tmp_path
    stage={'stage_id':'c1.lift','cycle_id':'c1','phase':'Lift','arm':'left','reference':{'start_time_sec':1.,'end_time_sec':2.}}
    _,progress,record=x.command(stage,.1,samples,SimpleNamespace(roles={'apple':[9,9,9]}),.1)
    assert progress==0 and record['reference_time_sec']==1. and np.allclose(goals[0][0],samples[0].tcp)
    assert issued[0][1].tolist()==[0.,.8] and issued[0][2]==[True,False]


def test_lift_endpoint_unknown_motion_or_empty_jaw_never_advances():
    x=RelativeLiftExecutor.__new__(RelativeLiftExecutor);x.holding=lambda *a:False
    with pytest.raises(SafetyError,match='empty/unknown'):x.refresh_lift_attachment({'phase':'Lift','object':'apple'},[],None)
    x.holding=lambda *a:True;x.comotion_record={'nonzero_evidence':False,'disagreement_m':0.}
    with pytest.raises(SafetyError,match='comotion unknown'):x.refresh_lift_attachment({'phase':'Lift','object':'apple'},[],None)


def test_real_empty_jaw_guard_from_v20_preserved(monkeypatch):
    x=RelativeLiftExecutor.__new__(RelativeLiftExecutor);x.accepted_grasp={'cycle':'c1','object':'apple','index':0,'aperture_m':.055,'object_TCP_distance_m':.04,'g_command':0.}
    x.stage={'cycle_id':'c1','phase':'Lift'};x.command_g=np.array([0.,1.]);x.T=np.eye(4);x.cfg={'grasp_region_margin_m':.05};x.visual_anchor=None;x.current_holding=True
    samples=[SimpleNamespace(tcp=np.zeros(3),aperture=.000326),SimpleNamespace(tcp=np.zeros(3),aperture=None)]
    import multitask_ordinary_runtime as base
    monkeypatch.setattr(base,'bounded_rgbd_capture',lambda owner,fn:(samples,SimpleNamespace(roles={'apple':np.array([.04,0,0])})))
    x.capture();assert x.current_holding is False and 'anomaly' in x.holding_source

def test_lift_to_move_rotated_endpoint_has_no_new_translation_or_rotation_jump():
    x=RelativeLiftExecutor.__new__(RelativeLiftExecutor);x.command_g=np.array([0.,1.]);x.holding=lambda *a:True
    x.comotion_record={'role':'apple','nonzero_evidence':True,'disagreement_m':.002}
    actual_R=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();demo_R=Rotation.from_euler('xyz',[-.2,.1,.05]).as_matrix()
    actual_p=np.array([.2,-.3,.15]);demo_p=np.array([.04,-.4,.1]);obj=actual_p+actual_R@np.array([.02,-.01,0])
    x.attachment={'a_actual':np.zeros(3),'a_ref':np.array([.015,-.02,0]),'R_demo_end':np.eye(3),'R_actual_grasp':np.eye(3)}
    x.b={'qcurve':lambda t:np.zeros((2,6))};x._shared_pose=lambda q,g,arm:(demo_p,demo_R)
    samples=[SimpleNamespace(q=np.ones(6),tcp=actual_p,rotation=actual_R),SimpleNamespace(q=np.ones(6),tcp=np.zeros(3),rotation=np.eye(3))]
    x._shared_pose=lambda q,g,arm:(actual_p,actual_R) if np.all(q==1) else (demo_p,demo_R)
    stage={'stage_id':'c1.lift','object':'apple','arm':'left','reference':{'end_time_sec':1.}}
    x.refresh_lift_attachment(stage,samples,SimpleNamespace(roles={'apple':obj}))
    goal,_=x.attached_goal((demo_p,demo_R),x.coarse_source)
    assert np.allclose(goal[0],actual_p) and np.allclose(goal[1],actual_R)
