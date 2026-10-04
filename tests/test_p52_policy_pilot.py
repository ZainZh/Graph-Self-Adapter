"""No simulator or pretrained network instantiated in these runner tests."""
from dataclasses import asdict
import json
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy
from tools.task_graph import p52_policy_pilot as pilot


def test_closed_loop_executes_neural_command_then_observes_feedback(tmp_path,monkeypatch):
    torch.manual_seed(52)
    config=Config(raw_dim=24,appearance_dim=8,role_dim=8,hidden=16,latent=4)
    model=StagePolicy(config);checkpoint=tmp_path/'head.pt'
    torch.save(dict(config=asdict(config),model=model.state_dict()),checkpoint)
    events=[]
    class Env:
        def _get_observations(self,**kwargs):return {}
        def _check_success(self):events.append('goal');return True
        def step(self,action):
            events.append('act');assert action.shape==(7,) and action[-1]==1
            return {},None,None,None
        def close(self):events.append('env_close')
    class Renderer:
        def __init__(self,*args):pass
        def observe(self,obs):events.append('observe');return dict(marker=np.zeros(1))
        def close(self):events.append('renderer_close')
    class Vision:
        binding={}
        def __init__(self,*args):pass
        def encode(self,path):
            events.append('vision')
            return dict(raw=torch.ones(2,3,24),boxes=torch.ones(2,3,4),visible=torch.ones(2,3),proprio=torch.ones(18))
        def close(self):events.append('vision_close')
    monkeypatch.setattr(pilot,'verify',lambda binding:None)
    monkeypatch.setattr(pilot,'initialize',lambda *args:Env())
    monkeypatch.setattr(pilot,'PublicRenderer',Renderer);monkeypatch.setattr(pilot,'Vision',Vision)
    monkeypatch.setattr(pilot,'snapshot',lambda *args:(dict(supervisor_dummy=np.zeros(1)),None))
    binding=dict(cells=[dict(index=0,entry=0,arm='online')],entries=[{}],environment={},public={},vision={},
                 checkpoints=dict(online=dict(path=str(checkpoint))),goal_hold_frames=3,max_actions=180)
    path=tmp_path/'freeze.json';path.write_text(json.dumps(binding));out=tmp_path/'run'
    result=pilot.run(path,0,out)
    assert result['actions']==2 and result['local_success'] and not result['full_task_success_evidence']
    assert events[:7]==['observe','vision','goal','act','observe','vision','goal']
    records=[json.loads(s) for s in (out/'public/ACTIONS.jsonl').read_text().splitlines()]
    assert [r['updates'] for r in records]==[0,1,2]
    assert [r['executed'] for r in records]==[True,True,False]
    assert all('official_success' not in r for r in records)
    assert (out/'SEAL.json').exists() and (out/'private/GOAL.jsonl').exists()
    assert events[-3:]==['renderer_close','env_close','vision_close']


def test_feature_boundary_rejects_extra_truth_and_nonfinite():
    import pytest
    config=Config()
    f=dict(raw=torch.zeros(2,3,3072),boxes=torch.zeros(2,3,4),visible=torch.ones(2,3),proprio=torch.zeros(18))
    pilot.validate_features(f,config)
    with pytest.raises(ValueError):pilot.validate_features(dict(f,world_book=torch.eye(4)),config)
    f['raw'][0,0,0]=float('nan')
    with pytest.raises(ValueError):pilot.validate_features(f,config)
