"""No simulator: validate matched skill loading and retained entry-failure scope."""
from dataclasses import asdict
import json
import torch
from prove.task_graph.p52_stage_policy import Config
from prove.task_graph.p53_short_insert import ShortInsertPolicy
from tools.task_graph import p53_micro_evaluate as micro


def test_blocked_code_entry_is_recorded_without_policy_or_physics(tmp_path,monkeypatch):
    binding=dict(cells=[dict(index=i,entry=0,arm=a) for i,a in enumerate(micro.ARMS)],entries=[dict(near_entry_qualified=False)])
    path=tmp_path/'freeze.json';path.write_text(json.dumps(binding))
    monkeypatch.setattr(micro,'verify',lambda b:None)
    monkeypatch.setattr(micro,'initialize',lambda *args:(_ for _ in ()).throw(AssertionError('No physics for rejected entry')))
    monkeypatch.setattr(micro,'load_skill',lambda *args:(_ for _ in ()).throw(AssertionError('No policy for rejected entry')))
    results=[micro.run(path,i,tmp_path/f'arm{i}') for i in range(6)]
    assert len(results)==6
    assert all(not r['micro_executed'] and not r['system_success'] and r['micro_success'] is None for r in results)
    assert all(r['actions']==0 and not r['official_only_hold3'] for r in results)


def test_short_checkpoint_reuses_same_frozen_runtime_architecture(tmp_path):
    config=Config(raw_dim=24,appearance_dim=8,role_dim=8,hidden=16,latent=4)
    source=ShortInsertPolicy(config);path=tmp_path/'synthetic_schema_checkpoint.pt'
    torch.save(dict(schema='p53_short_insert_skill_v01',model=source.state_dict(),config=asdict(config)),path)
    restored,_=micro.load_skill(path)
    assert all(torch.equal(source.state_dict()[k],restored.state_dict()[k]) for k in source.state_dict())
