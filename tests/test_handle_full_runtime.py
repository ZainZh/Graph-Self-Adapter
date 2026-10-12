import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'tools/deployment') not in sys.path:sys.path.insert(0,str(ROOT/'tools/deployment'))
from handle_full_runtime import HandleFullExecutor,static_cycle_cues
from handle_full_route_plan import graph_copy_with_handle_route,checked_local_qualification


def fixture(tmp_path):
    spec=importlib.util.spec_from_file_location('_full_handle_fixture',ROOT/'tests/test_object_centered_runtime.py');helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    old,robot,clock,stages,roles=helper.fixture(tmp_path)
    for stage in stages:stage.update(cycle_id='c03',object='white_mug',arm='right')
    for frame in old.b['frames']:frame['surface_centroids_world_m']['white_mug']=roles['white_mug'].tolist()
    config={**old.cfg,'hardware_reference_speed_scale':1.,'robot_pad_geometry_file':str(ROOT/'evaluations/sam_flat_pad_grasp_v09/full_prepared/ROBOT_FLAT_PAD_GEOMETRY.json')}
    rng=np.random.default_rng(5);tube=np.c_[rng.normal(.15,.0005,80),rng.normal(0,.0005,80),np.linspace(.04,.08,80)];cup=tube+[-.04,0,0];table=cup.copy();table[:,2]=0.
    vision=old.vision;vision.latest_observation_stamp=0.;vision.raw_camera=SimpleNamespace(last_physical_time=0.);vision.T=np.eye(4);vision.T[1,3]=-1.;vision.cloud=lambda role:table if role=='table_patch' else tube if role=='white_handle' else cup
    reground=[];vision.reground=lambda cycle:reground.append(cycle)
    ex=HandleFullExecutor(old.b,robot,vision,None,[],config,tmp_path,clock=clock,sleep=clock.sleep);ex.initial_roles={k:v.copy() for k,v in roles.items()};ex.stage=stages[0]
    return ex,robot,clock,stages,roles,reground


def test_white_initialization_sends_no_motion_and_preserves_static_cue_contract(tmp_path):
    ex,robot,clock,stages,roles,_=fixture(tmp_path);ex.initialize_white_cycle(stages[0],*ex.capture())
    assert not robot.commands and np.allclose(ex.static_source_env,0.)
    assert np.array_equal(ex.static_source_env,ex.code_static_source_handled)
    assert np.array_equal(ex.static_target_env,ex.code_static_target_handled)
    assert ex.handle_routes[0]['arm']=='right'


def test_right_actor_uses_same_handle_code_and_does_not_complete_at_prepoint(tmp_path):
    ex,robot,clock,stages,roles,_=fixture(tmp_path);before=robot.q.copy();target,progress,record=ex.command(stages[0],.1,*ex.capture(),.1)
    assert record['actor_only_flags']==[False,True] and np.array_equal(target[0],before[0]) and np.max(abs(target-before))<=.0090001
    plan=ex.handle_goals['c03'];robot.q[1,:3]=plan['pre'][0]
    from scipy.spatial.transform import Rotation
    robot.q[1,3:]=Rotation.from_matrix(plan['pre'][1]).as_rotvec();ex._last_cartesian_goal=plan['pre'];ex._goal_reached=True;ex.safety.last_target=robot.q.copy()
    facts=ex.predicates(stages[0],0.,robot.q.copy(),*ex.capture());assert facts['progress'] is False
    assert not ex.b['skill'].features # no hidden super.command NN/action for init


def test_second_white_cycle_requires_fresh_grounding_and_cannot_reuse_world_goal(tmp_path):
    ex,robot,clock,stages,roles,reground=fixture(tmp_path);ex.initialize_white_cycle(stages[0],*ex.capture())
    next_stage={**stages[0],'cycle_id':'c04','stage_id':'c04.approach'};ex._manipulated.add('white_mug');ex.stage=next_stage
    ex.initialize_white_cycle(next_stage,*ex.capture());assert reground==['c04'] and 'c04' in ex.handle_goals and not robot.commands


def test_route_preserves28node_order_and_rejects_unqualified_full_execution(tmp_path):
    original=json.loads((ROOT/'evaluations/xtrainer_laptop_v01/portable/cup/graph/task_graph.json').read_text());routed=graph_copy_with_handle_route(original)
    assert len(routed['stages'])==28 and [s['stage_id'] for s in routed['stages']]==[s['stage_id'] for s in original['stages']]
    assert sum(s.get('controller')=='code_handle_tube' for s in routed['stages'])==2
    result=tmp_path/'RESULT.json';result.write_text(json.dumps({'local_qualified':False}));freeze=tmp_path/'FREEZE.json';freeze.write_text('{}')
    with pytest.raises(ValueError,match='not passed'):checked_local_qualification(result,freeze)


def test_actual_handle_grasp_attachment_makes_next_lift_start_continuous(tmp_path):
    ex,robot,clock,stages,roles,_=fixture(tmp_path);ex.initialize_white_cycle(stages[0],*ex.capture())
    grasp=stages[1];ex.stage=grasp;actual=ex._shared_pose(robot.q,ex.command_g,'right');ex._last_cartesian_goal=actual;ex._goal_reached=True
    ex.register_attachment(grasp,*ex.capture())
    reference=ex._shared_pose(ex.b['qcurve'](grasp['reference']['end_time_sec']),ex.command_g,'right')
    next_goal,object_goal=ex.attached_goal(reference,ex.coarse_source)
    assert np.allclose(next_goal[0],actual[0]) and np.allclose(next_goal[1],actual[1])
    assert np.allclose(object_goal,roles['white_mug']) and not robot.commands


def test_qualified_gate_checks_exact_candidate_sources_and_pad_extractor_binding(tmp_path):
    import hashlib
    code=tmp_path/'code.py';code.write_text('robot_flat_geometry=True\n')
    driver=tmp_path/'driver.py';driver.write_text("VARIANT='test_only_fixture_variant'\n")
    artifact=tmp_path/'pad.json';artifact.write_text(json.dumps({'source_binding':{'path':str(code),'sha256':hashlib.sha256(code.read_bytes()).hexdigest()}}))
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps({'jobs':[{'candidate':4,'kind':'handle_local'}],'bindings':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (code,driver)]}))
    result=tmp_path/'result.json';result.write_text(json.dumps({'variant':'test_only_fixture_variant','candidate':4,'local_qualified':True,'post_seal_local_qualification':{'passed':True,'private_post_seal_only':True,'physical_window_sec':60}}))
    assert checked_local_qualification(result,freeze,candidate=4,code_source=code,driver_source=driver,pad_artifact=artifact)
    code.write_text('changed=True\n')
    with pytest.raises(ValueError,match='source changed'):checked_local_qualification(result,freeze,candidate=4,code_source=code,driver_source=driver,pad_artifact=artifact)


def test_full_vision_logs_stage_and_actual_camera_clock_without_private_pose(tmp_path,monkeypatch):
    import handle_full_runtime as runtime
    from prove.xtrainer.perception import Observation
    vision=runtime.HandleFullVision.__new__(runtime.HandleFullVision);vision.mask_records=[];vision.frame_count=0;vision.frame_records=[];vision.evidence_root=tmp_path
    vision.latest_masks={'white_mug':np.ones((2,2),bool)};vision.clock=lambda:13.;vision.raw_camera=SimpleNamespace(last_physical_time=7.5);vision.stage_context=lambda:{'stage_id':'c03.02.grasp','phase':'Grasp'}
    def capture(parent):parent.frame_count+=1;return Observation(12.,np.zeros((2,2,3),np.uint8),{},'fakecamera_only')
    monkeypatch.setattr(runtime.HandleVision,'capture',capture)
    vision.capture();record=vision.frame_records[-1]
    assert record['stage_id']=='c03.02.grasp' and record['physical_time_sec']==7.5 and record['observed_monotonic_sec']==12.
    vision.raw_camera.last_physical_time=None;vision.capture();assert vision.frame_records[-1]['physical_time_sec'] is None
    assert 'real camera' in vision.frame_records[-1]['clock_source'] and not any('qpos' in k for k in record)


def test_full_white_lift_uses_envelope_height_and_reference_relative_rotation(tmp_path):
    ex,robot,clock,stages,roles,_=fixture(tmp_path);ex.initialize_white_cycle(stages[0],*ex.capture())
    robot.q[1]=np.array([.15,0,.08,0,0,0]);ex.safety.last_target=None;ex.stage=stages[1];ex.register_attachment(stages[1],*ex.capture())
    original=ex.b['qcurve'];reference=original(0.).copy()
    def curve(t):
        q=reference.copy();q[1,3]+=.2*max(0.,min(1.,t-2.));return q
    ex.b['qcurve']=curve;lift=stages[2];ex.stage=lift
    _,_,record=ex.command(lift,.1,*ex.capture(),.1);goal=ex.handle_goals['c03']['lift_goal'];prior=ex.handle_goals['c03']['envelope_prior'];pad=np.array(prior['robot_pad_midpoint_TCP_local_m'])
    assert np.isclose((goal[0]+goal[1]@pad)[2],prior['goal_grasp_point_height_m'])
    demo=ex._shared_pose(curve(lift['reference']['end_time_sec']),ex.command_g,'right')
    expected=demo[1]@ex.attachment['R_demo_end'].T@ex.attachment['R_actual_grasp']
    assert np.allclose(goal[1],expected) and record['controller']=='functional_handle_flat_pad_envelope_Code'


def test_full_rgbd_retry_restores_rejected_history_and_marks_stale_evidence(tmp_path):
    from collections import deque
    from rgbd_capture_retry import bounded_rgbd_capture
    from prove.xtrainer.io import SafetyError
    owner=SimpleNamespace(rgb_history=deque([('prior',)],maxlen=3),vision=SimpleNamespace(latest_capture_stamp=1.,frame_records=[{'observed_monotonic_sec':1.}]),clock=lambda:1.3,output=tmp_path,cfg={'safety':{'max_age_sec':.25}})
    calls=[]
    def capture():
        calls.append(1);owner.rgb_history.append(('stale' if len(calls)==1 else 'new',))
        if len(calls)==1:raise SafetyError('stale real RGB-D')
        return 'fresh_feedback','new_image'
    assert bounded_rgbd_capture(owner,capture)==('fresh_feedback','new_image') and len(calls)==2
    assert ('stale',) not in owner.rgb_history and owner.vision.frame_records[0]['discarded_due_to_stale'] is True


def test_supported_grasp_can_request_root_despite_sparse_samples_and_wide_gap(tmp_path):
    ex,robot,clock,stages,roles,_=fixture(tmp_path);ex.initialize_white_cycle(stages[0],*ex.capture());stage=stages[1];ex.stage=stage
    plan=ex.handle_goals['c03'];plan['closed_at']=-10.;plan['goal']=ex._shared_pose(robot.q,ex.command_g,'right')
    samples,obs=ex.capture();samples[1].aperture=.03 # exceeds former partial-radius threshold; Root remains authoritative
    _,progress,record=ex.command(stage,.1,samples,obs,.1)
    assert progress==1. and 'no partial-radius' in record['Grasp_completion_gate']
