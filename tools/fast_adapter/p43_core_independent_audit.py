"""Audit exactly one completed frozen 96-cell independent model batch."""
import argparse,json,hashlib
import numpy as np
from pathlib import Path
import torch
from tools.fast_adapter.p43_core_audit import audit
from tools.fast_adapter.p43_validate_frozen import validate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',required=True);args=parser.parse_args()
    frozen=validate(args.freeze);root=Path(frozen['execution']['out']);expected=[];physical=set();seen=set()
    for seed,split in zip(frozen['seeds'],frozen['splits']):
        for arm in frozen['arms']:
            name=f'{split}_{seed}_{arm}';expected.append(name)
            if split not in seen:physical.add(name)
        seen.add(split)
    actual={p.parent.name for p in root.glob('*/RESULT.json')}
    if actual!=set(expected) or len(expected)!=96:raise ValueError('Independent batch is incomplete or contains extra cells; do not audit partial outcomes')
    torch.set_num_threads(2);rows=[]
    for name in expected:
        row=audit(root/name,frozen['execution']['checkpoint'],physics=name in physical)
        row['result_sha256']=hashlib.sha256((root/name/'RESULT.json').read_bytes()).hexdigest();rows.append(row)
    prefix_equal=[]
    for seed,split in zip(frozen['seeds'],frozen['splits']):
        actions=[]
        for arm in frozen['arms']:
            trace=np.load(root/f'{split}_{seed}_{arm}'/'TRACE.npy',allow_pickle=True)
            actions.append(np.array([np.r_[r['command_p'],r['command_R'].ravel(),r['closed']] for r in trace[:12]]))
        prefix_equal.append(np.array_equal(actions[0],actions[1]))
    record=dict(common_first12_actions_equal=all(prefix_equal),scope='Sealed independent replay, no new trials',cells=len(rows),passed=all(r['passed'] for r in rows) and all(prefix_equal),
        physics_cells=sorted(physical),physics_replays=sum(r['full_physics_replay'] for r in rows),audits=rows)
    (root/'AUDIT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='audits'}),flush=True)
    if not record['passed']:raise SystemExit(1)

if __name__=='__main__':main()
