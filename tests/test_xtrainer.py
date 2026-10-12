import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from prove.xtrainer.io import (NativeRPC,FeedbackStream,DashboardStop,SafetyLayer,SafetyError,sdk_dtype,
                              decode_feedback,load_calibration,dls_retarget,RobotSample,feedback_fk_check)
from prove.xtrainer.runtime import load_bundle,Executor
from prove.xtrainer.perception import Observation,role_geometry,LiveVision
from prove.xtrainer.agent import RealRootJudge,REQUEST_SCHEMA,REPLY_SCHEMA,APPROVAL_SCHEMA,digest,write
from prove.xtrainer.mock import MockClock,MockRobot,MockVision,MockJudge,MockStop
from tools.task_graph.xtrainer_rsi import check,main,validate_authorization,nominal_envelopes,REPO,EXTERNAL
CONFIG=REPO/'task_packages/xtrainer_v01/CONFIG_PROPOSED.json'


def test_construct_check_no_implicit_connect(monkeypatch):
    import socket
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('implicit hardware connection'))
    calls=[]
    rpc=NativeRPC(socket_factory=lambda:calls.append('connect'))
    stream=FeedbackStream('192.168.5.1',EXTERNAL/'include/xtrainer_clover/dobot_control/robots/dobot_api.py',socket_factory=lambda *a,**k:pytest.fail('connect'))
    stop=DashboardStop('192.168.5.1',socket_factory=lambda *a,**k:pytest.fail('connect'))
    report,cfg,bundle=check(CONFIG,'put_mug_on_box')
    assert report['hardware_connections']==0 and report['graph_stages']==28 and report['model_input_dim']==29
    assert rpc.sock is None and stream.sock is None and stop.sock is None and not calls


def test_packet_degree_mm_conversion_and_unknown_aperture():
    dtype=sdk_dtype(EXTERNAL/'include/xtrainer_clover/dobot_control/robots/dobot_api.py')
    raw=np.zeros(1,dtype=dtype);p=raw[0]
    for k,v in {'len':dtype.itemsize,'test_value':0x123456789abcdef,'q_actual':[180]*6,'qd_actual':[90]*6,
                'tool_vector_actual':[100,200,300,0,0,90],'robot_mode':5,'enable_status':1,'tool':1,
                'tool[6]':[0,0,197,0,0,0],'time_stamp':12}.items():p[k]=v
    stamp,sample=decode_feedback(raw.tobytes(),dtype,5.)
    assert stamp==12 and sample.aperture is None and np.allclose(sample.q,np.pi)
    assert np.allclose(sample.qdot,np.pi/2) and np.allclose(sample.tcp,[.1,.2,.3])
    assert np.allclose(sample.rotation,Rotation.from_euler('z',90,degrees=True).as_matrix())
    p['test_value']=0
    with pytest.raises(SafetyError):decode_feedback(raw.tobytes(),dtype,5.)


class RPCSocket:
    def __init__(self,result=None,fail=False):self.result=result;self.fail=fail;self.sent=[];self.endpoint=None
    def connect(self,endpoint):self.endpoint=endpoint
    def send(self,message):self.sent.append(pickle.loads(message))
    def recv(self):
        if self.fail:raise TimeoutError('synthetic timeout')
        return pickle.dumps(self.result)
    def close(self):pass


def test_native_rpc_pack_left_right_units_flags_and_errors():
    socket=RPCSocket();rpc=NativeRPC(socket_factory=lambda:socket);rpc.connect()
    q=np.arange(12).reshape(2,6)*.01;g=[0.,1.];rpc.command(q,g,[True,False])
    payload=socket.sent[-1]
    assert payload['method']=='command_joint_state' and payload['args']['flag_in']==[True,False]
    assert np.allclose(payload['args']['joint_state'],np.r_[q[0],0,q[1],1])
    socket.result={'__robot_remote_error__':True}
    with pytest.raises(SafetyError,match='remote'):rpc.command(q,g,[False,True])
    socket.fail=True
    with pytest.raises(SafetyError,match='timeout'):rpc.fk(q,g,'right')
    assert not rpc.connected


def test_calibration_transform_and_pose_preserving_dls():
    camera,T=load_calibration(EXTERNAL/'script/real_zed_collection/calibration/robot_camera_apriltag_left_global.yaml',
                             EXTERNAL/'script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml')
    assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
    class FK:
        def fk(self,q,g,arm):
            v=q[0 if arm=='left' else 1]
            return v[:3],Rotation.from_rotvec(v[3:]+np.array([v[0],0,0])).as_matrix()
    q=np.zeros((2,6));result=dls_retarget(FK(),q,[0,0],'left',[.01,0,0],np.eye(4))
    assert result[0,0]>.009 and abs(result[0,3]+result[0,0])<1e-4
    assert np.array_equal(result[1],q[1])


def test_actual_stale_mode_offset_tracking_safety():
    _,cfg,b=check(CONFIG,'put_mug_on_box');clock=MockClock();robot=MockRobot(b,clock);samples=robot.samples()
    layer=SafetyLayer(cfg['safety'],clock);layer.samples(samples)
    clock.t=1
    with pytest.raises(SafetyError,match='stale'):layer.samples(samples)
    for field,value in [('error',True),('tool',0),('running',True)]:
        sample=robot.samples()[0];setattr(sample,field,value)
        if field!='running':
            with pytest.raises(SafetyError):layer.samples([sample,robot.samples()[1]])


def test_metric_depth_live_resolution_and_median_pool():
    depth=np.ones((4,4));mask=np.ones((4,4),bool);K=np.array([[2,0,1.5],[0,2,1.5],[0,0,1.]])
    point=role_geometry(depth,{'cup':mask},K,np.eye(4))['cup']
    assert np.allclose(point,[0,0,1])
    with pytest.raises(SafetyError):role_geometry(depth,{'cup':np.ones((8,8))},K,np.eye(4))


def test_prompt_handoff_retains_same_frame_then_new_capture(tmp_path,monkeypatch):
    clock=MockClock();camera=SimpleNamespace(connect=lambda:None,close=lambda:None)
    images=[np.zeros((4,4,3),np.uint8),np.ones((4,4,3),np.uint8)]
    calls=[]
    def capture():
        rgb=images[min(len(calls),1)];calls.append(rgb.copy());return clock(),rgb,np.ones((4,4)),np.eye(3)
    camera.capture=capture
    cfg={'role_boxes':{},'prompt_frame_sha256':None,'prompt_frame_output':str(tmp_path/'first.png'),
         'prompt_reply_path':str(tmp_path/'prompts.json'),'prompt_wall_timeout_sec':3,'max_age_sec':.25}
    tracker=SimpleNamespace(initialize=lambda rgb,boxes:calls.append(('initialized',rgb.copy())),
                            track=lambda rgb:{'cup':np.ones((4,4),bool)})
    vision=LiveVision(camera,np.eye(4),cfg,clock=clock);monkeypatch.setattr(vision,'_tracker',lambda:tracker)
    def sleep(_):
        clock.sleep(.1);write(tmp_path/'prompts.json',{'frame_sha256':digest(tmp_path/'first.png'),'role_boxes':{'cup':[0,0,4,4]}})
    monkeypatch.setattr('prove.xtrainer.perception.time.sleep',sleep)
    obs=vision.capture()
    assert calls[1][0]=='initialized' and not calls[1][1].any() and obs.rgb.all()


def bridge_pair(tmp_path,*,change=None):
    clock=MockClock();clock.t=5.;sample=RobotSample(5.,np.zeros(6),np.zeros(6),np.zeros(3),np.eye(3),None,'actual_test')
    obs=Observation(5.,np.zeros((4,4,3),np.uint8),{'cup':np.zeros(3)},'actual_test')
    obs.history=[(4.8,obs.rgb,[]),(4.9,obs.rgb,[]),(5.,obs.rgb,[])]
    stage={'stage_id':'release','object':'cup'}
    def responder(_):
        clock.sleep(.1);status=json.loads((tmp_path/'STATUS.json').read_text());rp=Path(status['request']);request=json.loads(rp.read_text())
        reply={k:request[k] for k in ('request_id','task','stage_id','graph_sha256')}
        reply.update(schema=REPLY_SCHEMA,request_sha256=digest(rp),decision='complete',judged_facts={'released':True},
                     rationale='explicit test fixture',evidence_frame_ids=['f0','f1','f2'])
        if change=='unknown_true':
            reply['decision']='unknown';reply['judged_facts']={'released':True,'holding':True,'placement_complete':True}
        pp=rp.parent/'ROOT_REPLY.json';write(pp,reply)
        approval={'schema':APPROVAL_SCHEMA,'request_id':request['request_id'],'request_sha256':digest(rp),
                  'reply_sha256':digest(pp),'actual_root_identity':'root_test'}
        if change=='identity':approval['actual_root_identity']='fake'
        write(rp.parent/'ROOT_APPROVAL.json',approval)
    judge=RealRootJudge(tmp_path,task='cup',graph_sha='a'*64,identity='root_test',clock=clock,sleep=responder)
    def recapture():
        import copy
        new=copy.deepcopy(sample);new.stamp=clock();new.q[0]=.1 if change=='drift' else 0.
        newobs=Observation(clock(),obs.rgb,obs.roles,'fresh_test');return [new,new],newobs
    return judge,stage,([sample,sample],obs),recapture


@pytest.mark.parametrize('change',[None,'identity','drift'])
def test_real_bridge_identity_and_postreply_drift(tmp_path,change):
    judge,stage,before,recapture=bridge_pair(tmp_path,change=change)
    if change=='identity':
        with pytest.raises(SafetyError):judge.ask(stage,['released'],before=before,recapture=recapture)
    else:
        facts,after=judge.ask(stage,['released'],before=before,recapture=recapture)
        assert facts['released'] is (None if change=='drift' else True)
        assert judge.records[0]['postreply_rgb_stamp']>before[1].stamp


def test_wait_heartbeat_fails_before_any_reply(tmp_path):
    judge,stage,before,recapture=bridge_pair(tmp_path)
    def broken():raise SafetyError('feedback disabled during wait')
    with pytest.raises(SafetyError,match='disabled'):judge.ask(stage,['released'],before=before,recapture=recapture,heartbeat=broken)
    assert not list(tmp_path.rglob('ROOT_REPLY.json'))


def test_full_graph_mock_and_release_unknown_stops(tmp_path):
    result=main(['--mock','--output',str(tmp_path/'full')])
    assert result['graph_completed'] and len(result['stage_trace'])==28
    assert result['hardware_motion_commands']==0 and result['mock_command_count']>400
    assert any(row['controller']['controller']=='frozen_numpy_mlp' for row in result['trace'])
    assert result['latent_updates']==0
    report,cfg,b=check(CONFIG,'put_mug_on_box');clock=MockClock();robot=MockRobot(b,clock);vision=MockVision(robot,clock)
    vision.set_context(b['graph']['stages'][0],0,b);judge=MockJudge(vision,robot);stops=[MockStop(),MockStop()]
    cfg.update(mode='mock',T_left_from_right=np.eye(4).tolist(),known_initial_gripper_commands=b['initial_g'].tolist(),initial_gripper_state_confirmed=True)
    cfg['safety'].update(joint_min_rad=[-8]*6,joint_max_rad=[8]*6,workspace_min_m=[-8]*3,workspace_max_m=[8]*3,
                         max_tracking_rad=.03)
    original=judge.ask
    def unknown(stage,facts,**kw):
        values,after=original(stage,facts,**kw)
        if 'released' in facts:values['released']=None
        return values,after
    judge.ask=unknown
    result=Executor(b,robot,vision,judge,stops,cfg,tmp_path/'negative',clock=clock,sleep=clock.sleep).run()
    assert not result['graph_completed'] and 'UNKNOWN' in result['failure']
    assert all(s.calls==1 for s in stops) and any('release' in row['stage_id'] for row in result['trace'])


def test_model_tamper_and_scope_zero_no_connect(tmp_path):
    report,cfg,b=check(CONFIG,'put_mug_on_box')
    scope=tmp_path/'scope.json';scope.write_text(json.dumps({'hardware_motion_commands_authorized':False}))
    with pytest.raises(SafetyError,match='not authorized'):
        validate_authorization(scope,tmp_path/'absentfreeze',report,CONFIG,mode='execute',task='put_mug_on_box',output=tmp_path/'out')
    from unittest.mock import patch
    with patch('prove.xtrainer.runtime.digest',return_value='changed'):
        with pytest.raises(SafetyError,match='hash mismatch'):load_bundle()


def test_native_fk_validation_and_strict_stop_parse():
    samples=[RobotSample(0.,np.zeros(6),np.zeros(6),np.zeros(3),np.eye(3),None,'actual_test') for _ in range(2)]
    rpc=SimpleNamespace(fk=lambda *a:(np.zeros(3),np.eye(3)))
    assert feedback_fk_check(rpc,samples,[0,0])['passed']
    samples[1].tcp=np.array([.1,0,0]);assert not feedback_fk_check(rpc,samples,[0,0])['passed']
    for reply,valid in [(b'0,{},ResetRobot();',True),(b'0,{},EnableRobot();',False),(b'-1,{},ResetRobot();',False)]:
        stop=DashboardStop('test');stop.sock=SimpleNamespace(sendall=lambda b:None,recv=lambda n:reply)
        if valid:stop.stop()
        else:
            with pytest.raises(SafetyError):stop.stop()


def test_hardware_reference_scaling_preserves_limits_and_negative_envelope():
    report,cfg,bundle=check(CONFIG,'put_mug_on_box')
    assert all(row['hardware_scaled']['passed'] for row in report['nominal_joint_envelopes'])
    assert sum(not row['original']['passed'] for row in report['nominal_joint_envelopes'])>0
    unscaled=nominal_envelopes(bundle,{**cfg,'hardware_reference_speed_scale':1.})
    assert any(not row['hardware_scaled']['passed'] for row in unscaled)
    _,knife_cfg,knife=check(CONFIG,'put_knife_board_new')
    assert all(row['hardware_scaled']['passed'] for row in nominal_envelopes(knife,knife_cfg))


def test_static_overlap_expires_and_nonzero_comotion_is_required(tmp_path):
    _,cfg,b=check(CONFIG,'put_mug_on_box');clock=MockClock();robot=MockRobot(b,clock);vision=MockVision(robot,clock)
    stage=b['graph']['stages'][2];vision.set_context(stage,0,b)
    cfg.update(mode='mock',T_left_from_right=np.eye(4).tolist())
    runtime=Executor(b,robot,vision,MockJudge(),[MockStop(),MockStop()],cfg,tmp_path,clock=clock,sleep=clock.sleep)
    runtime.stage=stage
    tcp=robot.q[0,:3].copy();point=vision.capture().roles[stage['object']].copy()
    runtime.visual_anchor=(stage['object'],0,point-tcp,tcp,point,0.)
    clock.t=.3;runtime.capture()
    assert runtime.current_holding is None and runtime.holding_source=='static_overlap_is_unknown'
    vision.attachments[stage['object']]=(0,point-tcp);robot.q[0,0]+=.005;runtime.capture()
    assert runtime.current_holding is True and runtime.holding_source=='nonzero_visible_object_TCP_comotion'


def test_task_stage_and_root_request_budgets_are_fail_closed(tmp_path):
    _,cfg,b=check(CONFIG,'put_mug_on_box');clock=MockClock();robot=MockRobot(b,clock);vision=MockVision(robot,clock)
    runtime=Executor(b,robot,vision,MockJudge(),[MockStop(),MockStop()],cfg,tmp_path,clock=clock,sleep=clock.sleep)
    clock.t=cfg['max_task_wall_sec']+1
    with pytest.raises(SafetyError,match='task wall'):runtime.guard_limits()
    runtime.task_started=clock();runtime.stage_started=clock()-cfg['max_stage_wall_sec']-1
    with pytest.raises(SafetyError,match='stage wall'):runtime.guard_limits()
    runtime.stage_started=None;runtime.root_request_count=cfg['max_root_requests_per_task']
    with pytest.raises(SafetyError,match='request budget'):runtime.semantics(b['graph']['stages'][0],['holding'],None)


def test_unknown_decision_never_authorizes_true_reply_facts(tmp_path):
    judge,stage,before,recapture=bridge_pair(tmp_path,change='unknown_true')
    facts,after=judge.ask(stage,['released','holding','placement_complete'],before=before,recapture=recapture)
    assert facts=={'released':None,'holding':None,'placement_complete':None}
    record=judge.records[0]
    assert record['decision']=='unknown' and record['facts']==facts
    assert record['original_reply']['decision']=='unknown'
    assert record['original_reply']['judged_facts']=={'released':True,'holding':True,'placement_complete':True}
    saved=json.loads(next(tmp_path.rglob('ROOT_REPLY.json')).read_text())
    assert record['original_reply']==saved
    assert after[1].stamp>before[1].stamp
