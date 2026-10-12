"""Geometry and continuity only; all I/O below is explicitly synthetic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from prove.xtrainer.io import RobotSample
from prove.xtrainer.perception import Observation

root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('_object_centered_test',root/'tools/deployment/object_centered_runtime.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t
    def sleep(self,t):self.t+=t
class Robot:
    def __init__(self,clock):self.q=np.full((2,6),-.1);self.clock=clock;self.commands=[]
    def fk(self,q,g,arm):
        v=q[0 if arm=='left' else 1];return v[:3].copy(),Rotation.from_rotvec(v[3:]).as_matrix()
    def samples(self):
        return [RobotSample(self.clock(),q.copy(),np.zeros(6),q[:3].copy(),Rotation.from_rotvec(q[3:]).as_matrix(),None,'synthetic_fixture') for q in self.q]
    def command(self,q,g,flags):self.commands.append((np.asarray(q).copy(),flags));self.q=np.asarray(q).copy()
class Skill:
    def __init__(self):self.features=[]
    def command(self,features,**kw):
        self.features.append(np.asarray(features).copy());return SimpleNamespace(action=np.array([.02,0,0,0,0,0]),record={'controller':'synthetic_delta_q_reference'})


def fixture(tmp_path):
    clock=Clock();robot=Robot(clock);roles={'red_mug':np.array([.3,.2,.1]),'white_mug':np.array([.1,0,.1]),'box':np.array([.5,.4,.1])}
    rows=[{'id':'progress','predicate':'reference_progress'},{'id':'pose','predicate':'actor_command_reached'}]
    stages=[]
    for i,phase in enumerate(('Approach','Grasp','Lift','Move','Align','Release','Wait')):
        stages.append({'stage_id':'c01.'+phase.lower(),'cycle_id':'c01','phase':phase,'arm':'left','object':'red_mug','target':'box',
                       'controller':'neural_short' if phase in ('Grasp','Align') else 'code_reference_retarget',
                       'reference':{'start_time_sec':float(i),'end_time_sec':float(i+1)},'reference_speed_scale':1.,
                       'completion_predicates':rows,'invariant_predicates':[],'done':['progress','pose'],'invariants':[],'holding_contract':'none'})
    frames=[{'time_sec':float(i),'surface_centroids_world_m':{'red_mug':[.1,.1,.1],'box':[.2,.2,.1]},
             'tcp_position':[[.2,.2,.2],[.4,.4,.4]],'tcp_rotation':[np.eye(3).tolist()]*2} for i in range(8)]
    qref=np.array([[.2,.2,.2,.1,0,0],[.4,.4,.4,0,0,0]])
    bundle={'graph':{'stages':stages},'refs':{'handover_anchors':{}},'skill':Skill(),'frames':frames,'frame_times':np.arange(8.),
            'qcurve':lambda t:qref.copy(),'times':np.arange(8.),'gripper':np.ones((8,2)),'initial_q':np.zeros((2,6)),'initial_g':np.ones(2),'bindings':{}}
    cfg={'mode':'synthetic','safety':{'max_age_sec':.25,'max_velocity_rad_sec':1.,'max_step_rad':.04,'max_tracking_rad':.05,
          'joint_min_rad':[-2]*6,'joint_max_rad':[2]*6,'workspace_min_m':[-2]*3,'workspace_max_m':[2]*3},
         'T_left_from_right':np.eye(4).tolist(),'known_initial_gripper_commands':[1,1],'initial_gripper_state_confirmed':True,
         'hardware_reference_speed_scale':.1,'command_dt_sec':.1,'max_root_requests_per_task':128,'max_task_wall_sec':3600,'max_stage_wall_sec':600}
    vision=SimpleNamespace(capture=lambda:Observation(clock(),np.zeros((4,4,3),np.uint8),roles,'synthetic_fixture'))
    ex=m.ObjectCenteredExecutor(bundle,robot,vision,None,[],cfg,tmp_path,clock=clock,sleep=clock.sleep)
    return ex,robot,clock,stages,roles


def test_current_pose_start_and_actor_only_approach_step(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.start();ex.stage=stages[0];samples,obs=ex.capture()
    before=robot.q.copy();target,progress,record=ex.command(stages[0],.1,samples,obs,.1)
    assert progress==0. and np.max(abs(target-before))<=.01000001
    assert robot.commands[-1][1]==[True,False] and np.array_equal(target[1],before[1])
    assert np.allclose(record['goal_TCP_shared_m'],[.4,.3,.2])
    facts=ex.predicates(stages[0],progress,target,*ex.capture());assert facts['pose'] is False and facts['progress'] is False


def test_neural_delta_is_reference_FK_correction_not_actual_q_increment(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[0];ex.command(stages[0],.1,*ex.capture(),.1)
    ex.stage=stages[1];before=robot.q.copy();target,_,record=ex.command(stages[1],.1,*ex.capture(),.1)
    assert np.allclose(record['goal_TCP_shared_m'],[.42,.3,.2])
    assert np.max(abs(target-before))<=.01000001 and not np.allclose(target[0],before[0]+[.02,0,0,0,0,0])
    assert np.allclose(ex.b['skill'].features[-1][12:18],0.)


def test_reference_clock_does_not_advance_until_prior_full_pose_reached(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[0];ex.command(stages[0],.1,*ex.capture(),.1)
    ex.stage=stages[2];_,progress_a,record_a=ex.command(stages[2],.1,*ex.capture(),.1)
    _,progress_b,record_b=ex.command(stages[2],.2,*ex.capture(),.1)
    assert progress_a==progress_b and record_a['reference_time_sec']==record_b['reference_time_sec']


def test_move_end_align_release_coarse_target_is_continuous(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[0];ex.command(stages[0],.1,*ex.capture(),.1)
    move=stages[3];ex._phase=move['stage_id'];ex._phase_elapsed=10.;ex._last_cartesian_goal=None;ex.stage=move
    _,_,move_record=ex.command(move,10.,*ex.capture(),.1)
    for stage in (stages[4],stages[5],stages[6]):
        if stage['phase']=='Wait':ex._empty_hand_anchor=ex._shared_pose(robot.q,ex.command_g,'left') # explicit completed Release fixture
        ex.stage=stage;_,_,record=ex.command(stage,.1,*ex.capture(),.1)
        assert np.allclose(record['coarse_world_translation_m'],move_record['coarse_world_translation_m'])


def test_hard_command_budget_and_completed_scope_stop_priority(tmp_path):
    import pytest
    from prove.xtrainer.io import SafetyError
    ex,robot,clock,stages,roles=fixture(tmp_path)
    ex.cfg['maximum_hardware_commands']=2;ex.command_index=2
    with pytest.raises(SafetyError,match='command budget'):ex.guard_limits()
    ex.engineering_first_stage_only=True;ex._scope_complete=True
    with pytest.raises(SafetyError,match='engineering_scope_stop'):ex.guard_limits()


def test_live_surface_change_does_not_rewrite_cycle_static_environment_feature(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[0];ex.command(stages[0],.1,*ex.capture(),.1)
    # Motion/visibility error is not an original frozen static environment feature.
    roles['red_mug']=roles['red_mug']+np.array([.002,0,0]);ex.stage=stages[1]
    _,_,record=ex.command(stages[1],.1,*ex.capture(),.1)
    assert record['local_source_known'] and np.allclose(record['local_source_residual_m'],0.)
    assert np.allclose(ex.b['skill'].features[-1][12:15],0.)


def test_grasp_bootstrap_is_bounded_fresh_sensor_evidence_without_microstep_root_request(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[2]
    tcp=robot.q[0,:3].copy();point=roles['red_mug'].copy();ex.visual_anchor=('red_mug',0,point-tcp,tcp,point,0.)
    ex._grasp_bootstrap=('red_mug',0.);clock.t=1.
    samples,obs=ex.capture();assert ex.current_holding is True and ex.holding_source=='bounded_visual_grasp_bootstrap'
    assert ex.semantics(stages[2],['holding'],(samples,obs))==(samples,obs)
    clock.t=11.;ex.capture();assert ex.current_holding is None


def test_command_intent_persists_before_partial_rpc_failure(tmp_path):
    import json,pytest
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[0]
    def partial(*args):raise RuntimeError('ServoJ sent; gripper failed')
    robot.command=partial
    with pytest.raises(RuntimeError,match='gripper failed'):ex.command(stages[0],.1,*ex.capture(),.1)
    intent=json.loads((tmp_path/'COMMAND_INTENT.json').read_text())
    assert intent['attempt_count']==1 and intent['acknowledged_count']==0
    assert intent['status']=='attempted_not_acknowledged' and intent['partial_motion_possible_without_ack']


def test_start_view_pairing_zero_nominal_shift_and_no_occluded_end_relock(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path)
    ex.b['frames'][0]['surface_centroids_world_m']['red_mug']=roles['red_mug'].tolist()
    ex.b['frames'][1]['surface_centroids_world_m']['red_mug']=(roles['red_mug']+np.array([.00962,0,0])).tolist()
    ex.stage=stages[0];target,progress,record=ex.command(stages[0],.1,*ex.capture(),.1)
    assert np.allclose(ex.coarse_source,0.) and np.allclose(record['locked_source_translation_m'],0.)
    robot.q[0]=np.array([.2,.2,.2,.1,0,0]);ex.safety.last_target=robot.q.copy();roles['red_mug']+=np.array([.012,0,0])
    facts=ex.predicates(stages[0],1.,robot.q.copy(),*ex.capture())
    assert facts['progress'] is True and np.allclose(ex.coarse_source,0.)


def test_surface_median_shift_is_not_a_hard_scene_failure(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[0];ex.command(stages[0],.1,*ex.capture(),.1)
    roles['red_mug']+=np.array([.012,0,0])
    ex.command(stages[0],.2,*ex.capture(),.1)
    assert len(robot.commands)==2


def registered_attachment_fixture(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path)
    ex.stage=stages[0];ex.command(stages[0],.1,*ex.capture(),.1)
    ex.coarse_target=np.array([.3,.2,0.]);ex.target_anchor=roles['box'].copy()
    robot.q[0]=np.array([.45,.25,.2,.1,0,0]);ex.safety.last_target=robot.q.copy()
    ex.command_g=np.array([0.,1.]);roles['red_mug']=np.array([.31,.20,.10])
    ex.register_attachment(stages[1],robot.samples(),ex.vision.capture())
    return ex,robot,clock,stages,roles


def test_actual_grasp_to_first_lift_goal_is_continuous_at_exact_reference_start(tmp_path):
    ex,robot,clock,stages,roles=registered_attachment_fixture(tmp_path)
    ex.stage=stages[2];_,progress,record=ex.command(stages[2],.1,*ex.capture(),.1)
    assert record['reference_time_sec']==stages[2]['reference']['start_time_sec']
    assert np.allclose(record['goal_TCP_shared_m'],[.45,.25,.2],atol=1e-10)
    assert np.linalg.norm(np.asarray(record['Cartesian_error_before_command']))<1e-10


def test_actual_attachment_corrects_TCP_not_object_landing_target(tmp_path):
    ex,robot,clock,stages,roles=registered_attachment_fixture(tmp_path)
    nominal=ex._shared_pose(ex.b['qcurve'](4.),ex.command_g,'left')
    goal,object_goal=ex.attached_goal(nominal,ex.coarse_target)
    expected_object=nominal[0]+nominal[1]@ex.attachment['a_ref']+ex.coarse_target
    assert np.allclose(object_goal,expected_object)
    assert np.allclose(goal[0]+goal[1]@ex.attachment['a_actual'],expected_object)
    old_object=object_goal.copy();ex.attachment['a_actual']+=np.array([.04,0,0])
    changed_goal,changed_object=ex.attached_goal(nominal,ex.coarse_target)
    assert np.allclose(changed_object,old_object) and not np.allclose(changed_goal[0],goal[0])


def test_align_learned_actual_pose_is_held_at_release_then_empty_wait_relative_entry(tmp_path):
    ex,robot,clock,stages,roles=registered_attachment_fixture(tmp_path)
    # Actual Align endpoint includes prior learned/attachment correction, not nominal .2 TCP.
    robot.q[0]=np.array([.51,.36,.27,.15,.02,0]);ex.safety.last_target=robot.q.copy()
    ex.stage=stages[5];_,_,record=ex.command(stages[5],.1,*ex.capture(),.1)
    assert np.allclose(record['goal_TCP_shared_m'],[.51,.36,.27])
    ex._empty_hand_anchor=ex._release_pose_anchor;ex.stage=stages[6]
    _,_,wait=ex.command(stages[6],.1,*ex.capture(),.1)
    assert np.allclose(wait['goal_TCP_shared_m'],[.51,.36,.27])


def test_centroid_changes_three_to_eight_cm_do_not_change_NN_static_cues(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.initial_roles={r:p.copy() for r,p in roles.items()}
    ex.stage=stages[0];ex.command(stages[0],.1,*ex.capture(),.1)
    ex.stage=stages[1];_,_,first=ex.command(stages[1],.1,*ex.capture(),.1);feature=ex.b['skill'].features[-1][12:18].copy()
    target_shift=ex.coarse_target.copy()
    roles['red_mug']+=np.array([.03,-.04,.08]);roles['box']+=np.array([.08,.05,.03])
    _,_,second=ex.command(stages[1],.2,*ex.capture(),.1)
    assert np.array_equal(ex.b['skill'].features[-1][12:18],feature)
    assert np.array_equal(ex.coarse_target,target_shift)
    assert 'scene initial' in second['target_environment_prior']


def test_Code_handled_static_shift_is_not_given_again_to_frozen_NN(tmp_path):
    from prove.xtrainer.portable import load_portable_bundle
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.stage=stages[0]
    _,_,record=ex.command(stages[0],.1,*ex.capture(),.1)
    assert np.linalg.norm(ex.static_source_env)>0 and np.array_equal(ex.static_source_env,ex.code_static_source_handled)
    ex.stage=stages[1];ex.command(stages[1],.1,*ex.capture(),.1)
    frozen=load_portable_bundle('evaluations/xtrainer_laptop_v01/portable/cup')['skill']
    features=ex.b['skill'].features[-1]
    step=frozen.command(features,current_surface_xyz=None,reference_surface_xyz=None,phase='Grasp',holding=None)
    assert np.array_equal(features[12:18],np.zeros(6)) and np.array_equal(step.action,np.zeros(6))


def test_approach_keeps_actor_open_until_actual_entry_then_grasp_profile_closes(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path)
    ex.b['gripper'][0]=[.99,.97];ex.b['gripper'][1]=[.702,.95];ex.b['gripper'][2]=[0.,.95]
    ex.command_g=np.array([.99,.97]);sent=[]
    original=robot.command
    def capture_command(q,g,flags):sent.append(np.asarray(g).copy());original(q,g,flags)
    robot.command=capture_command;ex.stage=stages[0]
    target,progress,record=ex.command(stages[0],.1,*ex.capture(),.1)
    assert progress==0. and sent[-1][0]==.99 and sent[-1][1]==.97
    robot.q[0]=np.array([.4,.3,.2,.1,0,0]);ex.safety.last_target=robot.q.copy()
    ready=ex.predicates(stages[0],1.,robot.q.copy(),*ex.capture());assert ready['progress'] and ready['pose']
    ex.stage=stages[1];ex.command(stages[1],.1,*ex.capture(),.1)
    assert sent[-1][0]==.702 and sent[-1][1]==.97


def test_paired_static_prior_uses_same_measurement_method_without_double_compensation(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path)
    ex.initial_roles={k:v.copy() for k,v in roles.items()}
    ex.static_reference_points={k:v.copy() for k,v in roles.items()}
    ex.stage=stages[0];_,_,record=ex.command(stages[0],.1,*ex.capture(),.1)
    assert np.allclose(record['locked_source_translation_m'],0.) and np.allclose(record['locked_target_translation_m'],0.)
    roles['red_mug']+=.02;ex.stage=stages[1];ex.command(stages[1],.1,*ex.capture(),.1)
    assert np.allclose(ex.b['skill'].features[-1][12:18],0.)


def test_paired_prior_only_replaces_initial_unmanipulated_static_reference(tmp_path):
    ex,robot,clock,stages,roles=fixture(tmp_path);ex.initial_roles={k:v.copy() for k,v in roles.items()}
    ex.static_reference_points={k:v.copy() for k,v in roles.items()};ex._manipulated.add('red_mug')
    ex.stage=stages[0];_,_,record=ex.command(stages[0],.1,*ex.capture(),.1)
    declared=np.asarray(ex.nearest_ref(0.)['surface_centroids_world_m']['red_mug'])
    assert np.allclose(record['locked_source_translation_m'],roles['red_mug']-declared)
