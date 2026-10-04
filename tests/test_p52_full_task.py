"""Original-task integration checks without MuJoCo or pretrained inference."""
from types import SimpleNamespace
import json
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy,StageRuntime
from prove.task_graph.executor import JudgeResult
from prove.task_graph.libero_supervisor import SupervisorReport
from tools.task_graph import p52_full_task as full


def test_structural_judge_cannot_report_dropped_goal_success():
    judge=full.InsertJudge()
    for i in range(5):
        r=judge.observe(True,(False,False),False)
        assert not r['complete']
        assert r['stop']==('book_contact_lost_5' if i==4 else None)
    assert r['goal_hold']==0


def test_single_sided_unknown_is_logged_and_can_hold_native_goal():
    judge=full.InsertJudge()
    for i in range(3):
        r=judge.observe(True,(True,False),False)
        assert r['attachment']=='UNKNOWN_single_sided'
        assert r['complete']==(i==2)
    assert full.InsertJudge().observe(True,(True,True),True)['stop']=='wrong_target'


def test_prefix_exits_at_s6_without_old_s7_backend_or_transition(tmp_path,monkeypatch):
    graph=SimpleNamespace(nodes=[SimpleNamespace(id=f's{i}',skill='Move') for i in range(1,8)])
    graph.stage=lambda ident:graph.nodes[int(ident[1:])-1]
    calls=[]
    class Skill:
        revision='fake';segments={f's{i}' for i in range(1,8)};diagnostics={}
        def __init__(self,*args):pass
        def set_observation(self,obs):pass
        def on_enter(self,stage):calls.append(stage)
        def on_exit(self):pass
    class Router:
        def required_kind(self,*args):return 'code'
        def register(self,*args,**kwargs):pass
    class Executor:
        current_stage='s1'
        def __init__(self,*args,**kwargs):pass
        def tick(self,stage,*args,**kwargs):
            self.current_stage=f's{int(stage.id[1:])+1}'
            return JudgeResult('complete',stage.id,'termination_verified',next_stage=self.current_stage)
    class Supervisor:
        def __init__(self,*args,**kwargs):pass
        def observe(self,*args):return SupervisorReport({},False,None,{})
    class Env:
        def _get_observations(self,**kwargs):return {}
        def step(self,action):raise AssertionError('No extra entry-transition actions permitted')
    monkeypatch.setattr(full.TaskGraph,'from_dict',lambda *args:graph)
    monkeypatch.setattr(full,'MixedController',Skill);monkeypatch.setattr(full,'MixedRouter',Router)
    monkeypatch.setattr(full,'GraphExecutor',Executor);monkeypatch.setattr(full,'LiberoSupervisor',Supervisor)
    monkeypatch.setattr(full,'grounding',lambda graph,stage,*args:stage)
    monkeypatch.setattr(full,'insert_entry_domain',lambda skill:(True,{}))
    monkeypatch.setattr(full,'snapshot',lambda *args:({'supervisor_marker':np.zeros(1)},None))
    calibration=tmp_path/'calibration.json';calibration.write_text('{}')
    out=tmp_path/'prefix';out.mkdir()
    binding=dict(prefix=dict(runtime_graph={},checkpoint='unused',calibration=str(calibration)),environment=dict(source='unused'),prefix_max_actions=400)
    result=full.run_prefix(Env(),binding,out,'test')
    assert result['qualified'] and result['actions']==0
    assert calls==[f's{i}' for i in range(1,7)]


def test_insert_public_model_actions_no_terminal_truth_input(tmp_path,monkeypatch):
    torch.manual_seed(52)
    cfg=Config(raw_dim=24,appearance_dim=8,role_dim=8,hidden=16,latent=4)
    runtime=StageRuntime(StagePolicy(cfg),online=True)
    features=dict(raw=torch.zeros(2,3,24),boxes=torch.zeros(2,3,4),visible=torch.ones(2,3),proprio=torch.zeros(18))
    class Env:
        t=0
        def _get_observations(self,**kwargs):return {}
        def step(self,a):
            assert a.shape==(7,) and a[-1]==1
            self.t+=1;return {},0,False,{}
    class Vision:
        def encode(self,path):return {k:v.clone() for k,v in features.items()}
    class Renderer:
        def observe(self,obs):return dict(public_marker=np.zeros(1))
    env=Env()
    monkeypatch.setattr(full,'structural_measurement',lambda *args:(True,(True,False),False))
    monkeypatch.setattr(full,'snapshot',lambda *args:({'supervisor_marker':np.zeros(1)},None))
    for folder in ('public','private'):(tmp_path/folder).mkdir()
    result=full.run_insert(env,dict(insert_max_actions=180),runtime,Vision(),Renderer(),tmp_path,{})
    assert result['graph_complete'] and result['actions']==2 and result['updates']==2
    records=[json.loads(line) for line in (tmp_path/'public/ACTIONS.jsonl').read_text().splitlines()]
    assert [r['executed'] for r in records]==[True,True,False]
    assert all('official_success' not in r and 'finger_contacts' not in r for r in records)
    assert len(list((tmp_path/'private').glob('frame_*.npz')))==3
