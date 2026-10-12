"""Backend boundary contracts only: no physics stepping, GPU, camera or robot."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
from scipy.spatial.transform import Rotation
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(relative,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sim=load('tools/deployment/sim_robot_io.py','_sim_io_contract')
entry=load('tools/deployment/shared_cup_entry.py','_shared_entry_contract')


def test_fk_only_changes_independent_robot_kinematics_scratch():
    robot=sim.SimRobot.__new__(sim.SimRobot);robot.fk_lock=threading.Lock();robot.inverse=np.eye(4)
    scratch=SimpleNamespace(qpos=np.zeros(14))
    scratch.site=lambda arm:SimpleNamespace(xpos=scratch.qpos[:3].copy(),xmat=np.eye(3).reshape(9))
    robot.scratch=scratch
    class MainData:
        def __getattr__(self,name):pytest.fail('main physical state accessed during FK')
    robot.world=SimpleNamespace(model=SimpleNamespace(qpos0=np.zeros(14),numeric=lambda n:SimpleNamespace(data=[.02])),
                                arm_qpos_indices=np.arange(12),fingers=[],data=MainData())
    robot.mj=SimpleNamespace(mj_forward=lambda *a:None)
    position,_=robot.fk(np.ones((2,6)),np.ones(2),'left')
    assert np.array_equal(position,[1,1,1])


def test_camera_renders_only_RGB_depth_never_segmentation_or_hidden_roles():
    robot=SimpleNamespace(lock=threading.RLock())
    robot.world=SimpleNamespace(model=object(),data=SimpleNamespace(qpos=np.zeros(4),qvel=np.zeros(4),mocap_pos=np.empty((0,3)),mocap_quat=np.empty((0,4)),time=1.5))
    def snapshot(model):return SimpleNamespace(qpos=np.zeros(4),qvel=np.zeros(4),mocap_pos=np.empty((0,3)),mocap_quat=np.empty((0,4)),time=0.)
    class Renderer:
        def __init__(self,*a,**kw):self.depth=False
        def disable_segmentation_rendering(self):pass
        def enable_segmentation_rendering(self):pytest.fail('GT segmentation invoked')
        def disable_depth_rendering(self):self.depth=False
        def enable_depth_rendering(self):self.depth=True
        def update_scene(self,*a,**kw):pass
        def render(self):return np.ones((4,6)) if self.depth else np.zeros((4,6,3),np.uint8)
    robot.mj=SimpleNamespace(Renderer=Renderer,MjData=snapshot,mj_forward=lambda *a:None)
    class Calibration:
        def _camera(self):return np.eye(3),np.eye(4)
        def _render(self):pytest.fail('frozen GT render invoked')
    robot.physics=SimpleNamespace(VisibleRGBD=Calibration)
    camera=sim.SimRGBDCamera(robot,width=6,height=4);camera.connect();stamp,rgb,depth,K=camera.capture()
    assert rgb.shape==(4,6,3) and depth.shape==(4,6) and camera.last_physical_time==1.5


def test_execute_without_freeze_refused_before_backend_start(tmp_path):
    with pytest.raises(Exception,match='scope/freeze'):
        entry.guard(None,None,{},mode='sim',output=tmp_path,runtime_sha='a'*64)


def test_public_sampler_exposes_only_actual_robot_feedback():
    robot=sim.SimRobot.__new__(sim.SimRobot);robot.error=None;robot.lock=threading.RLock();robot.inverse=np.eye(4)
    prop=SimpleNamespace(joint_pos=np.zeros((2,6)),joint_velocity=np.zeros((2,6)),tcp_position=np.zeros((2,3)),
                         tcp_rotation=np.array([np.eye(3),np.eye(3)]),finger_qpos=np.zeros((2,2)))
    robot.world=SimpleNamespace(proprioception=lambda:prop,model=SimpleNamespace(numeric=lambda n:SimpleNamespace(data=[.02])))
    samples=robot.samples()
    assert samples[0].aperture==.04 and 'proxy_not_hardware' in samples[0].source
    assert not hasattr(samples[0],'object_pose') and not hasattr(samples[0],'contact')


def test_root_wait_does_not_acquire_or_stop_physics_lock():
    # Holding is an actuator target, not a time pause. Public intent updates alone
    # never call stop_event or restore state. Test verifies actual simulated IO.
    robot=sim.SimRobot.__new__(sim.SimRobot);robot.lock=threading.RLock();robot.q_target=np.zeros((2,6));robot.g_target=np.ones(2);robot.commands=0
    class NoPause:
        def set(self):pytest.fail('physical time paused by command/hold')
    robot.stop_event=NoPause();robot.world=SimpleNamespace(proprioception=lambda:SimpleNamespace(joint_pos=np.zeros((2,6))))
    robot.command(np.ones((2,6))*.1,np.ones(2),[True,False]);robot.hold('left')
    assert robot.commands==1 and np.all(robot.q_target[0]==0) and np.all(robot.q_target[1]==0)


def test_changed_runtime_or_unpassed_simulation_refuses_hardware_before_IO(tmp_path):
    import json,hashlib
    scope=tmp_path/'scope.json';scope.write_text(json.dumps({'execution_authorized':True,'task':'put_mug_on_box','mode':'real','hardware_motion_commands_authorized':True}))
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps({'schema':'graph-self-adapter.shared-cup-freeze.v1','status':'result_free_frozen',
        'scope_sha256':hashlib.sha256(scope.read_bytes()).hexdigest(),'runtime_sha256':'other'}))
    with pytest.raises(Exception,match='RuntimeHash'):entry.guard(scope,freeze,{},mode='real',output=tmp_path/'future',runtime_sha='a'*64)


def test_shared_start_does_not_invent_gripper_confirmation_and_budget_is_clock_independent(tmp_path):
    import sys
    helper_spec=importlib.util.spec_from_file_location('_same_runtime_fixture',ROOT/'tests/test_object_centered_runtime.py')
    helper=importlib.util.module_from_spec(helper_spec);sys.modules[helper_spec.name]=helper;helper_spec.loader.exec_module(helper)
    ex,robot,clock,stages,roles=helper.fixture(tmp_path)
    shared=load('tools/deployment/shared_cup_runtime.py','_shared_v02_runtime')
    cfg=dict(ex.cfg);cfg['initial_gripper_state_confirmed']=False;cfg['hardware_reference_speed_scale']=1.
    candidate=shared.SharedCupExecutor(ex.b,robot,ex.vision,None,[],cfg,tmp_path,clock=clock,sleep=clock.sleep)
    candidate.start()
    assert candidate.cfg['initial_gripper_state_confirmed'] is False
    assert candidate.initial_state_diagnostics['actual_aperture_m']==[None,None]
    assert candidate.cfg['maximum_node_control_updates']==1000 and candidate.cfg['maximum_hardware_commands']==5000


def test_root_node_grasp_state_overrides_surface_negative_without_claiming_contact(tmp_path):
    import sys
    hs=importlib.util.spec_from_file_location('_node_state_fixture',ROOT/'tests/test_object_centered_runtime.py')
    helper=importlib.util.module_from_spec(hs);sys.modules[hs.name]=helper;hs.loader.exec_module(helper)
    ex,robot,clock,stages,roles=helper.fixture(tmp_path);shared=load('tools/deployment/shared_cup_runtime.py','_shared_v03_node_state')
    cfg=dict(ex.cfg);cfg['hardware_reference_speed_scale']=1.
    candidate=shared.SharedCupExecutor(ex.b,robot,ex.vision,None,[],cfg,tmp_path,clock=clock,sleep=clock.sleep)
    candidate.stage=stages[2];candidate.command_g=np.array([0.,1.]);point=roles['red_mug'];tcp=robot.q[0,:3]
    candidate.visual_anchor=('red_mug',0,point-tcp,tcp.copy(),point.copy(),0.)
    candidate.root_node_state={'role':'red_mug','holding':True,'support_permission':None,'certified_at':0.,
        'initial_object_TCP_delta':point-tcp,'certified_aperture_m':None}
    # Place toy object inside declared grip region; a 12mm median mismatch alone is not negative grip truth.
    roles['red_mug']=tcp+np.array([.04,.012,0]);clock.t=2.
    samples,obs=candidate.capture();assert candidate.current_holding is True
    assert 'not_measured_contact' in candidate.holding_source
    assert candidate.semantics(stages[2],['holding'],(samples,obs))==(samples,obs)
    candidate.root_node_state['holding']=None;candidate.capture();assert candidate.current_holding is None


def shared_retry_fixture(tmp_path,answers):
    import sys
    hs=importlib.util.spec_from_file_location('_node_retry_fixture',ROOT/'tests/test_object_centered_runtime.py')
    helper=importlib.util.module_from_spec(hs);sys.modules[hs.name]=helper;hs.loader.exec_module(helper)
    ex,robot,clock,stages,roles=helper.fixture(tmp_path)
    shared=load('tools/deployment/shared_cup_runtime.py','_shared_v06_retry')
    cfg=dict(ex.cfg);cfg['hardware_reference_speed_scale']=1.
    class Judge:
        def __init__(self):self.records=[]
        def ask(self,stage,facts,*,before,recapture,heartbeat=None):
            value=answers[min(len(self.records),len(answers)-1)];self.records.append({'test_fixture_only':True})
            return {fact:value for fact in facts},recapture()
    judge=Judge();candidate=shared.SharedCupExecutor(ex.b,robot,ex.vision,judge,[],cfg,tmp_path,clock=clock,sleep=clock.sleep)
    stages[1]['holding_contract']='bilateral_contact'
    candidate.stage=stages[1]
    return candidate,robot,clock,stages,roles,judge,helper


def test_same_node_unknown_reasks_fresh_without_sending_motion_then_certifies(tmp_path):
    candidate,robot,clock,stages,roles,judge,_=shared_retry_fixture(tmp_path,[None,True])
    after=candidate.semantics(stages[1],['holding'],candidate.capture())
    assert len(judge.records)==2 and not robot.commands
    assert candidate.root_node_state['holding'] is True and candidate.unknown_counts[(stages[1]['stage_id'],('holding',))]==1


def test_same_node_remaining_unknown_never_advances(tmp_path):
    candidate,robot,clock,stages,roles,judge,_=shared_retry_fixture(tmp_path,[None])
    with pytest.raises(Exception,match='UNKNOWN'):candidate.semantics(stages[1],['holding'],candidate.capture())
    assert len(judge.records)==3 and not robot.commands and candidate.root_node_state is None


def test_root_confirmed_lift_reregisters_visible_attachment_once_with_move_continuity(tmp_path):
    candidate,robot,clock,stages,roles,judge,helper=shared_retry_fixture(tmp_path,[True])
    original,_,_,_,_=helper.registered_attachment_fixture(tmp_path/'reference')
    candidate.attachment={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.attachment.items()}
    original_ref=candidate.attachment['a_ref'].copy();original_rotation=candidate.attachment['R_actual_grasp'].copy()
    candidate.coarse_target=original.coarse_target.copy();candidate.static_source_env=np.array([.01,0,0]);candidate.code_static_source_handled=candidate.static_source_env.copy()
    demo=candidate._shared_pose(candidate.b['qcurve'](stages[2]['reference']['end_time_sec']),candidate.command_g,'left')
    robot.q[0,:3]=[.45,.25,.2];robot.q[0,3:]=Rotation.from_matrix(demo[1]@candidate.attachment['R_demo_end'].T@candidate.attachment['R_actual_grasp']).as_rotvec()
    roles['red_mug']=np.array([.31,.2,.087]);candidate.refresh_lift_attachment(stages[2],robot.samples(),candidate.vision.capture())
    goal,obj=candidate.attached_goal(demo,candidate.coarse_source)
    assert np.allclose(goal[0],robot.q[0,:3]) and np.allclose(obj,roles['red_mug'])
    assert np.array_equal(candidate.attachment['a_ref'],original_ref) and np.array_equal(candidate.attachment['R_actual_grasp'],original_rotation)
    assert np.array_equal(candidate.static_source_env,[.01,0,0])
    candidate.refresh_lift_attachment(stages[2],robot.samples(),candidate.vision.capture());assert len(candidate.attachment_refresh_log)==1
