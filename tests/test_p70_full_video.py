import json
import numpy as np
import pytest
from tools.task_graph.p70_full_video import collect,sha

def test_single_handoff_timestamp_and_no_missing_frames(tmp_path):
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n');return p
    public=dict(eef_pos=np.zeros(3),eef_quat_xyzw=np.array([0,0,0,1]),gripper_qpos=np.zeros(2))
    for name in ['prefix/public/frame_0000.npz','prefix/public/frame_0001.npz','s9/public/frame_0000.npz','s9/public/frame_0001.npz']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,**public)
    trace=tmp_path/'prefix/controller/TRACE.jsonl';trace.parent.mkdir();trace.write_text(''.join(json.dumps(dict(t=t,public=f'public/frame_{t:04d}.npz',stage='near_entry'))+'\n' for t in [0,1]))
    predictions=tmp_path/'s9/public/PREDICTIONS.jsonl';predictions.write_text(''.join(json.dumps(dict(t=t,observation_sha256=sha(tmp_path/f's9/public/frame_{t:04d}.npz'),telemetry=dict(event='update' if t==1 else None)))+'\n' for t in [0,1]))
    seal=write('SEAL.json',dict(files={str(p.relative_to(tmp_path)):sha(p) for p in tmp_path.rglob('*') if p.is_file()}))
    result=dict(graph_complete=False,full_task_success_evidence=True,stage_qualification_only=False,seal_sha256=sha(seal),original_initial_restores=1,later_state_restores=0,prefix=dict(actions=1),s9=dict(actions=1),total_actions=2)
    write('RESULT.json',result)
    _,frames=collect(tmp_path)
    assert len(frames)==3 and frames[1]['controller']=='P68 FastAdapter' and frames[2]['update']
    result['total_actions']=3;write('RESULT.json',result)
    with pytest.raises(ValueError,match='timeline'):collect(tmp_path)

    result['total_actions']=2;result['stage_qualification_only']=True;write('RESULT.json',result)
    with pytest.raises(ValueError,match='Full-task evidence'):collect(tmp_path)
    result['stage_qualification_only']=False;result['full_task_success_evidence']=False;write('RESULT.json',result)
    with pytest.raises(ValueError,match='Full-task evidence'):collect(tmp_path)

    result['full_task_success_evidence']=True;result['total_actions']=3
    result['placement_verification']=dict(actions=1,graph_complete=True);result['graph_complete']=True
    path=tmp_path/'placement_verification/public/frame_0000.npz';path.parent.mkdir(parents=True);np.savez(path,**public)
    write('placement_verification/actions/VERIFY_ACTIONS.jsonl',dict(t=0,global_t=3,executed=True,public='public/frame_0000.npz',public_sha256=sha(path),complete=True,native_hold3=True))
    seal=write('SEAL.json',dict(files={str(p.relative_to(tmp_path)):sha(p) for p in tmp_path.rglob('*') if p.is_file() and p.name not in ('SEAL.json','RESULT.json')}));result['seal_sha256']=sha(seal);write('RESULT.json',result)
    _,frames=collect(tmp_path)
    assert len(frames)==4 and frames[-1]['t']==3 and frames[-1]['stage']=='placement_verification'
