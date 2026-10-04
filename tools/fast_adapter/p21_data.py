"""Replay P20's opened training conditions once and cache three frozen frontends."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.insert3d import Insert3D, domain, start_position, nominal_action, oracle_action, PROBES
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL
from prove.fast_adapter.lila_frontend import ThreeFrontends, FRONTENDS
from tools.fast_adapter.p20_data import query_samples
from tools.fast_adapter.p18_run import save_json, digest
R = Path('evaluations/fast_adapter_p21_frontends_v01')

def collect(shard, n=256, dino=DEFAULT_MODEL, smoke=False):
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    out = R/('engineering' if smoke else 'bank')/f'shard{shard}'
    out.mkdir(parents=True, exist_ok=False)
    env=Insert3D(); encoder=ThreeFrontends(dino)
    pending=[]; boxes=[]; features={k:[] for k in FRONTENDS}
    all_rgb=[]; all_boxes=[]; positions=[]; visibility=[]; actions=[]
    bases=[]; teachers=[]; query_actions=[]; query_states=[]; physics=[]
    started=time.time()
    def observe():
        rgb, box, visible = env.observe()
        pending.append(rgb); boxes.append(box)
        all_rgb.append(rgb); all_boxes.append(box)
        positions.append(env.state()); visibility.append(visible)
    def encode():
        values=encoder.encode_all(np.stack(pending), np.stack(boxes))
        for k,v in values.items():features[k].append(v.cpu().half())
        pending.clear(); boxes.clear()
    for i in range(n):
        sid=shard*256+i
        M,b,fr=domain(2026092400+sid); env.set_domain(M,b,fr)
        env.reset(start_position(2027192400+sid)); observe()
        for a in PROBES:env.step(a); observe()
        bs=[]; ts=[]; qs=[]; states=[]
        rng=np.random.default_rng(2028092400+sid)
        for pos,t in query_samples(2029092400+sid):
            env.reset(pos); observe()
            base=nominal_action(pos,t); target=oracle_action(pos,t,M,b)
            action=np.clip(target+rng.uniform(-.2,.2,3),-1,1)
            env.step(action); observe()
            bs.append(base); ts.append(target); qs.append(action); states.append(pos)
        bases.append(bs);teachers.append(ts);query_actions.append(qs);query_states.append(states)
        physics.append({'id':sid,'seed':2026092400+sid,'M':M.tolist(),'bias':b.tolist(),'friction':fr})
        if len(pending)>=112:encode()
        if (i+1)%32==0:print(json.dumps({'shard':shard,'domains':i+1,'seconds':time.time()-started}),flush=True)
    if pending:encode()
    for k,v in features.items():
        torch.save(torch.cat(v).reshape(n,14,*v[0].shape[1:]),out/f'{k}.pt')
    meta={'base':torch.tensor(np.array(bases),dtype=torch.float32),
          'teacher':torch.tensor(np.array(teachers),dtype=torch.float32),
          'query_action':torch.tensor(np.array(query_actions),dtype=torch.float32),
          'states_diagnostic_only':torch.tensor(np.array(query_states),dtype=torch.float32),
          'positions_diagnostic_only':torch.tensor(np.array(positions).reshape(n,14,3),dtype=torch.float64),
          'physics_evaluator_only':physics}
    torch.save(meta,out/'META.pt')
    np.savez_compressed(out/'RGB.npz',rgb=np.stack(all_rgb),boxes=np.stack(all_boxes))
    save_json(out/'RECEIPT.json',{'shard':shard,'domains':n,'images':n*14,
       'seconds':time.time()-started,'three_frontends_same_rgb':True,
       'min_visible_pixels':np.array(visibility).min(0).tolist(),
       'source_training_domain_ids':[shard*256,shard*256+n-1],
       'artifacts':[{ 'path':str(p),'sha256':digest(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    env.close();print('DATA_COMPLETE',shard,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True)
    p.add_argument('--domains',type=int,default=256);p.add_argument('--dino',default=DEFAULT_MODEL)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    collect(a.shard,a.domains,a.dino,a.smoke)
