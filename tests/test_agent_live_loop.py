"""No simulator construction: real file handoff/state invariants and pure gates."""
import json
from pathlib import Path
import numpy as np
import pytest
from prove.task_graph.agent_live_loop import PauseBroker, HandoffError, atomic_json
from prove.task_graph.agent_stage_judge import REPLY_SCHEMA, digest
from prove.task_graph.agent_live_adapter import semantic_contract, PERMISSION, ALIGN_READY, PLACEMENT
from tools.task_graph.twin_agent_live_rsi import validate_live_execution

STAGE = {'stage_id':'c01.align','phase':'Align','actor':'left_gripper','object':'cup','target':'box',
         'goal_relation':'supported','completion_predicates':[]}


def fixture_broker(tmp_path, decisions=('complete',), mutation=None):
    state = {'t':5., 'hash':'unchanged', 'counter':0, 'wall':0., 'holds':[], 'request_ids':[]}
    def capture(): return [(state['t'], np.zeros((8,10,3),dtype=np.uint8), 'main')], {}
    def respond(_):
        state['wall'] += .1
        status = json.loads((tmp_path/'STATUS.json').read_text()); folder = Path(status['request_path']).parent
        req = json.loads((folder/'REQUEST.json').read_text()); state['request_ids'].append(req['request_id'])
        decision = decisions[min(len(state['request_ids'])-1,len(decisions)-1)]
        reply = {k:req[k] for k in ('request_id','task','stage_id','graph_sha256','observed_physical_timestamp_sec')}
        reply.update(schema=REPLY_SCHEMA,request_sha256=digest(folder/'REQUEST.json'),decision=decision,
                     judged_facts={PERMISSION:True},rationale='test-only public-image handoff',evidence_frame_ids=['f0'])
        atomic_json(folder/'ROOT_REPLY.json',reply)
        approval={'schema':'graph-self-adapter.root-approval.v1','request_id':req['request_id'],
                  'request_sha256':digest(folder/'REQUEST.json'),'reply_sha256':digest(folder/'ROOT_REPLY.json'),
                  'actual_root_reply_identity':'root_test_fixture'}
        if mutation: mutation(state,reply,approval,folder)
        atomic_json(folder/'ROOT_APPROVAL.json',approval)
    def hold(d): state['holds'].append(d); state['t'] += d
    broker=PauseBroker(tmp_path,task='cup',graph_sha256='a'*64,root_identity='root_test_fixture',
                       wall_timeout_sec=1.,clock=lambda:state['wall'],sleeper=respond)
    args=dict(capture=capture,state_hash=lambda:state['hash'],physical_time=lambda:state['t'],hold=hold,
              counters=lambda:{'initialization_count':1,'restore_count':state['counter']})
    return broker,state,args


def test_actual_file_bridge_pause_identity_and_seal(tmp_path):
    broker,state,args=fixture_broker(tmp_path)
    result=broker.ask(STAGE,[PERMISSION],**args)
    assert result['actions'] is None and result['decision']=='complete'
    assert state['t']==5 and state['holds']==[]
    rec=broker.records[0]
    assert rec['pause_before_sha256']==rec['pause_after_sha256']=='unchanged'
    assert rec['actual_root_reply_identity']=='root_test_fixture'
    path=tmp_path/'seal.json'; broker.seal(path,bindings={'graph':'a'*64},counters={'init':1})
    assert json.loads(path.read_text())['sealed_before_private_evaluator'] is True
    with pytest.raises(FileExistsError): broker.seal(path,bindings={},counters={})


def test_unknown_two_holds_new_request_and_expiry(tmp_path):
    broker,state,args=fixture_broker(tmp_path,('unknown','unknown','complete'))
    assert broker.ask(STAGE,[PERMISSION],**args)['decision']=='complete'
    assert state['holds']==[.25,.25] and state['t']==5.5
    assert len(set(state['request_ids']))==3
    for rec in broker.records:
        assert rec['physical_time_sec']==rec['resume_physical_time_sec']
        assert rec['pause_before_sha256']==rec['pause_after_sha256']


def test_unknown_exhaustion_and_budget_fail_closed(tmp_path):
    broker,state,args=fixture_broker(tmp_path,('unknown',))
    with pytest.raises(HandoffError,match='exhausted'): broker.ask(STAGE,[PERMISSION],**args)
    assert len(state['holds'])==2
    broker.sequence=16
    with pytest.raises(HandoffError,match='budget'): broker.ask(STAGE,[PERMISSION],**args)


@pytest.mark.parametrize('mutation',['identity','hash','stage','state','counter'])
def test_invalid_messages_or_paused_state_change_fail_closed(tmp_path,mutation):
    def mutate(state,reply,approval,folder):
        if mutation=='identity': approval['actual_root_reply_identity']='impostor'
        if mutation=='hash': approval['reply_sha256']='b'*64
        if mutation=='stage':
            reply['stage_id']='other';atomic_json(folder/'ROOT_REPLY.json',reply)
            approval['reply_sha256']=digest(folder/'ROOT_REPLY.json')
        if mutation=='state': state['hash']='changed'
        if mutation=='counter': state['counter']=1
    broker,state,args=fixture_broker(tmp_path,mutation=mutate)
    with pytest.raises(HandoffError): broker.ask(STAGE,[PERMISSION],**args)
    assert state['holds']==[]
    assert json.loads((tmp_path/'STATUS.json').read_text())['state']=='fail_closed'


def test_wall_timeout_no_physics_or_hold(tmp_path):
    broker,state,args=fixture_broker(tmp_path)
    broker.sleeper=lambda _:state.update(wall=state['wall']+.2)
    with pytest.raises(HandoffError,match='timeout'):broker.ask(STAGE,[PERMISSION],**args)
    assert state['t']==5 and not state['holds']


def stage(phase):
    rows=[{'id':'progress','predicate':'reference_progress'},{'id':'robot','predicate':'actor_command_reached'}]
    if phase=='Release': rows += [{'id':'release','predicate':'released'},{'id':'goal','predicate':'stage_goal'}]
    return {**STAGE,'phase':phase,'completion_predicates':rows}


def test_semantic_cannot_skip_robot_or_actual_release():
    calls=[]
    def ask(s,ids):
        calls.append(ids);return {'decision':'complete','judged_facts':{ALIGN_READY:True,PLACEMENT:True}}
    facts={'progress':False,'robot':True,'release':True,'goal':None}
    out,_=semantic_contract(stage('Release'),facts,[],ask=ask)
    assert out['semantic.goal'] is None and not calls
    facts.update(progress=True,release=False)
    out,_=semantic_contract(stage('Release'),facts,[],ask=ask)
    assert out['release'] is False and not calls
    facts.update(release=True)
    out,_=semantic_contract(stage('Release'),facts,[],ask=ask)
    assert out['goal'] is True and out['release'] is True and calls==[[PLACEMENT]]


def test_continue_does_not_mark_complete_and_unrelated_gates_unchanged():
    original={'progress':True,'robot':True,'finite':True,'holding':False}
    ask=lambda s,i:{'decision':'continue','judged_facts':{ALIGN_READY:True}}
    out,_=semantic_contract(stage('Align'),original,[],ask=ask)
    assert out['semantic.goal'] is False and out['holding'] is False and out['finite'] is True
    out,_=semantic_contract(stage('Move'),original,[],ask=lambda *a:pytest.fail('Agent called during carry'))
    assert out==original


def test_result_free_scope_and_exact_binding_output_guard(tmp_path):
    source=tmp_path/'source.py';source.write_text('frozen')
    scope=tmp_path/'scope.json';scope.write_text(json.dumps({'execution_authorized':True,'tasks':['cup'],'condition_seed':51005,'trials':[0],'maximum_full_task_episodes':2}))
    freeze=tmp_path/'freeze.json'; output=tmp_path/'out'
    data={'schema':'graph-self-adapter.agent-live-freeze.v1','execution_variant':'agent_live_v06',
          'status':'result_free_frozen','scope_sha256':digest(scope),'bindings':[{'path':str(source),'sha256':digest(source)}],
          'jobs':[{'task':'cup','trial':0,'seed':51005,'fixed_z':True,'native_initial_state':True,
                   'output':str(output),'actual_root_reply_identity':'root'}]}
    freeze.write_text(json.dumps(data))
    kwargs=dict(task='cup',output=output,root_identity='root',required_paths=[source])
    assert validate_live_execution(scope,freeze,**kwargs)
    for field,value in [('fixed_z',False),('seed',92005),('native_initial_state',False),('actual_root_reply_identity','stub')]:
        bad=json.loads(json.dumps(data));bad['jobs'][0][field]=value;freeze.write_text(json.dumps(bad))
        with pytest.raises(ValueError):validate_live_execution(scope,freeze,**kwargs)
    freeze.write_text(json.dumps(data));source.write_text('changed')
    with pytest.raises(ValueError):validate_live_execution(scope,freeze,**kwargs)


def test_world_interface_pause_resume_and_hold_clock_without_physics(tmp_path,monkeypatch):
    """Fake world is test-only; production still requires actual Root reply files."""
    import sys
    from types import SimpleNamespace
    import prove.task_graph.agent_live_adapter as adapter
    import tools.task_graph.twin_agent_live_rsi as entry
    from tools.task_graph.twin_sensor_rsi import load_frozen_runner
    from prove.task_graph.twin_sensor_judge import GapCalibration
    class Renderer:
        def __init__(self,*args,**kwargs):self.view=None
        def disable_depth_rendering(self):pass
        def disable_segmentation_rendering(self):pass
        def update_scene(self,data,camera):self.view=camera
        def render(self):return np.full((8,10,3),2 if self.view=='overview' else 1,dtype=np.uint8)
        def close(self):pass
    def getstate(model,data,state,signature):state[:]=[data.time,float(data.ctrl[0])]
    fake_mujoco=SimpleNamespace(Renderer=Renderer,mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1),
                               mj_stateSize=lambda *a:2,mj_getState=getstate)
    monkeypatch.setitem(sys.modules,'mujoco',fake_mujoco)
    class FakeWorld:
        def __init__(self,*args,**kwargs):
            self.data=SimpleNamespace(time=0.,ctrl=np.zeros(1),xfrc_applied=np.zeros(2),qfrc_applied=np.zeros(2))
            self.model=object();self.height=8;self.width=10;self.camera='main';self._state_writes_after_initialization=0
            self.sensor_stage=None;self.sensor_runtime=None;self.sensor_unknown_since=None;self.sensor_motion_blocked=False
            cal=GapCalibration(.04,(1.,-1.),.04,0.,'fake-test-calibration')
            self.sensor_calibrations={'left':cal,'right':cal};self._previous_gripper_command=np.array([0.,0.])
            self._previous_joint_command=np.zeros((2,6));self.closed=False
        def initialize(self,q,g):self._set_robot_state(q,g);return self.proprioception()
        def _set_robot_state(self,q,g):self.q=np.asarray(q).copy();self._previous_gripper_command=np.asarray(g).copy()
        def proprioception(self):return SimpleNamespace(joint_pos=self.q,joint_velocity=np.zeros((2,6)),
                tcp_position=np.zeros((2,3)),tcp_rotation=np.tile(np.eye(3),(2,1,1)),finger_qpos=np.zeros((2,2)))
        def observe_visible(self,*a,**kw):return SimpleNamespace(physical_time_sec=self.data.time,rgb=np.ones((8,10,3),dtype=np.uint8))
        def _sensor_facts(self,*a,**kw):return {'bilateral_contact':True,'support_transition_candidate':None}
        def _servo(self,*a,**kw):pass
        def step_command_callback(self,cb,duration,*a,**kw):
            q,g,v=cb(self.data.time);self._servo(q,g,v);self.q=np.asarray(q).copy()
            self.data.time+=duration;return SimpleNamespace(physical_time_sec=self.data.time)
        def numerical_safety(self):return {'simulation_finite':True}
        def close(self):self.closed=True
    physics=SimpleNamespace(TwinWorld=FakeWorld)
    monkeypatch.setattr(adapter,'transition_world_class',lambda *a,**kw:FakeWorld)
    runner,snapshot,packages=load_frozen_runner('put_mug_on_box')
    monkeypatch.setattr(runner,'_physics',lambda:physics)
    worlds=entry.install_runtime(runner,task='put_mug_on_box',packages=packages,config=None,output=tmp_path,
                                root_identity='root_test_fixture',bindings={})
    world=runner._physics().TwinWorld();world.initialize(np.zeros((2,6)),np.zeros(2))
    world.sensor_stage={**STAGE,'holding_contract':'held_or_supported'}
    world.observe_visible();frames,_=world.live_capture()
    assert [x[2] for x in frames]==['main','overview']
    assert frames[0][1][0,0,0]==1 and frames[1][1][0,0,0]==2
    first_hash=world.live_hash();world.live_hold(.25)
    assert world.data.time==.25 and first_hash!=world.live_hash()
    assert world.live_counters()=={'initialization_count':1,'robot_initial_state_write_count':1,
                                  'restore_after_initialization_count':0,'external_force_write_count':0}
    clock=runner.StageReferenceClock(0.,1.,1.)
    assert clock.at(.25)==0. and clock.at(.35)==pytest.approx(.1)
    assert clock.progress(.35)==pytest.approx(.1)
    with pytest.raises(HandoffError,match='multiple initialization'):world.initialize(np.zeros((2,6)),np.zeros(2))
    world.close();assert world.closed
