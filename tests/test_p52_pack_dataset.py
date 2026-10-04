import json
import numpy as np
import pytest
import torch
from tools.task_graph.p52_pack_dataset import pack,sha,verify_trajectory


def make_trajectory(tmp_path):
    root=tmp_path/'native';(root/'public').mkdir(parents=True);(root/'private').mkdir()
    (root/'BINDING.json').write_text('{}')
    sources=[]
    for t in range(17):
        frame=root/'public'/f'frame_{t:04d}.npz';np.savez(frame,public_marker=t)
        sources.append(dict(path=str(frame),sha256=sha(frame)))
    transitions=[dict(t=t,stage_id='Insert',episode_id='job00',observation_sha256=sources[t]['sha256'],
                      next_observation_sha256=sources[t+1]['sha256'],executed_action=[0]*6+[1]) for t in range(16)]
    labels=[dict(t=t,public_observation_sha256=sources[t]['sha256'],expert_action=[0]*6+[1]) for t in range(17)]
    for path,rows in [(root/'public/TRANSITIONS.jsonl',transitions),(root/'private/LABELS.jsonl',labels)]:
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    seal=dict(binding_sha256=sha(root/'BINDING.json'),actions=16,observations=17,
              files={str(p.relative_to(root)):sha(p) for folder in ['public','private'] for p in (root/folder).iterdir()})
    (root/'SEAL.json').write_text(json.dumps(seal))
    (root/'RESULT.json').write_text(json.dumps(dict(seal_sha256=sha(root/'SEAL.json'),teacher_qualified=True)))
    cache=tmp_path/'features.pt';features=dict(raw=torch.zeros(17,2,3,3072),boxes=torch.zeros(17,2,3,4),visible=torch.ones(17,2,3),proprio=torch.zeros(17,18))
    torch.save(dict(schema='p52_public_features_v01',features=features,sources=sources,frontend={}),cache)
    return root,dict(episodes=[dict(episode_id='job00',split='train',directory=str(root),feature_cache=str(cache))])


def test_actual_seal_and_source_files_checked(tmp_path):
    root,manifest=make_trajectory(tmp_path)
    assert len(pack(manifest)['episodes'])==1
    np.savez(root/'public/frame_0000.npz',public_marker=999)
    with pytest.raises(ValueError,match='Sealed trajectory file changed'):pack(manifest)


def test_failed_teacher_cannot_enter_training(tmp_path):
    root,manifest=make_trajectory(tmp_path)
    path=root/'RESULT.json';r=json.loads(path.read_text());r['teacher_qualified']=False;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='qualified teacher'):pack(manifest)


def test_result_must_bind_actual_seal(tmp_path):
    root,_=make_trajectory(tmp_path)
    path=root/'SEAL.json';s=json.loads(path.read_text());s['actions']=15;path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='actual trajectory seal'):verify_trajectory(root)
