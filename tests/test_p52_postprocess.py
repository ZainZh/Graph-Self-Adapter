import json
from pathlib import Path
import numpy as np
from tools.task_graph import p52_postprocess as pipeline


def test_fixed_split_all_failures_retained_and_training_gate(tmp_path,monkeypatch):
    conditions=[dict(index=i,condition_seed=5200+i,split='train' if i<18 else 'validation') for i in range(24)]
    binding=tmp_path/'binding.json';binding.write_text(json.dumps(dict(conditions=conditions)))
    queue=tmp_path/'queue'
    for i in range(24):
        job=queue/f'worker_{i%6}'/f'job_{i:02d}';root=job/'episode'
        (root/'public').mkdir(parents=True);(root/'private').mkdir()
        (job/'ATTEMPT_RESULT.json').write_text(json.dumps(dict(job=dict(id=f'job_{i:02d}',seed=5200+i))))
        (root/'RESULT.json').write_text(json.dumps(dict(job_id=f'job_{i:02d}',split=conditions[i]['split'],prefix_qualified=True,teacher_qualified=i>=6)))
        (root/'SEAL.json').write_text('{}')
        np.savez(root/'private/frame_0000.npz',supervisor_sim_state=np.zeros(1),supervisor_ctrl=np.zeros(1))
        (root/'public/frame_0000.npz').write_bytes(b'synthetic format stub; extraction not invoked')
        (root/'public/TRANSITIONS.jsonl').write_text(''.join(json.dumps(dict(episode_id=f'job{i}'))+'\n' for _ in range(16)))
    monkeypatch.setattr(pipeline,'verify_trajectory',lambda path,**kwargs:({},{}))
    dino=tmp_path/'dino';lila=tmp_path/'lila';dino.mkdir();(lila/'models').mkdir(parents=True)
    for path in [dino/'model.safetensors',dino/'config.json',dino/'preprocessor_config.json',lila/'VISUAL_WEIGHTS.pt',lila/'models/vla_model_fm.py']:
        path.write_bytes(b'test binding only')
    cfg=tmp_path/'cfg.json';cfg.write_text('{}')
    args=(binding,[queue],tmp_path/'post','python','python',dino,lila,cfg,cfg)
    plan=pipeline.prepare(*args)
    assert len(plan['rows'])==24 and len(plan['feature_jobs'])==24
    assert plan['gate']==dict(ready=True,qualified_train=12,required_train=12,validation_entries=6,required_validation=6)
    assert len([e for e in plan['episodes'] if e['split']=='train'])==12
    assert [r['split'] for r in plan['rows']]==[c['split'] for c in conditions]
    # One further failure changes eligibility, never split/seed or denominator.
    path=queue/'worker_0/job_06/episode/RESULT.json';path.write_text(json.dumps(dict(job_id='job_06',split='train',prefix_qualified=True,teacher_qualified=False)))
    again=pipeline.prepare(binding,[queue],tmp_path/'post2','python','python',dino,lila,cfg,cfg)
    assert not again['gate']['ready'] and again['gate']['qualified_train']==11
    assert len(again['rows'])==24
