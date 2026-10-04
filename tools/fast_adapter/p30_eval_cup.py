"""Small P30 closed-loop development evaluation for the cup task."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.object_interaction_env import ObjectEnv, object_config
from prove.fast_adapter.object_interaction_adapter import ObjectAdapter
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL
from prove.fast_adapter.online_latent import OnlineLatent
from tools.fast_adapter.p29_common import Visual, load_base, task_root

ROOT=Path('evaluations/fast_adapter_p30_online_v01')

def run(seed, split, model_seed, n=8, horizon=200, correction_scale=0.5):
    base=load_base('mug-handle-lift'); blob=torch.load(ROOT/'models_continuous'/'mug-handle-lift'/f'seed{model_seed}'/'gradient.pt',weights_only=True)
    net=ObjectAdapter().cuda(); net.load_state_dict(blob['state']); net.eval().requires_grad_(False)
    visual=Visual(DEFAULT_MODEL,'mug-handle-lift',torch.load(task_root('mug-handle-lift')/'data/PCA.pt',weights_only=True))
    rows=[]
    for i in range(n):
        cfg=object_config('mug-handle-lift',3021000+(0 if split=='train' else 100)+i,split)
        reset=3022000+(0 if split=='train' else 100)+i
        def episode(adapter):
            env=ObjectEnv(cfg,render=True); env.reset(reset); previous=torch.zeros(1,4,device='cuda'); state=None; successes=[]
            for t in range(horizon):
                image=np.stack([env.rgb()]); v=visual.encode(image); p=torch.as_tensor(env.public()[None],device='cuda')
                with torch.no_grad(): ba=base(v,p,previous)
                if adapter is None:
                    with torch.no_grad(): action=ba
                else:
                    if state is None: state=OnlineLatent(adapter,1,window=8,correction_scale=correction_scale,warmup_steps=8)
                    action=state.act(v,p,previous,ba)
                action=action.detach(); env.step(action[0].cpu().numpy()); actual=action
                if adapter is not None:
                    nv=visual.encode(np.stack([env.rgb()])); np_=torch.as_tensor(env.public()[None],device='cuda')
                    state.observe(nv,np_,actual)
                previous=actual
                successes.append(bool(any(x['success'] for x in env.metrics)))
            result=any(successes); steps=next((j+1 for j,x in enumerate(successes) if x),None); env.close(); return result,steps,state.updates if state else 0
        b,bs,_=episode(None); a,ss,updates=episode(net)
        rows.append({'split':split,'index':i,'base':b,'gradient':a,'base_first_success':bs,'gradient_first_success':ss,'gradient_updates':updates})
        print(split,i,'base',b,'gradient',a,'updates',updates,flush=True)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=8);p.add_argument('--scale',type=float,default=0.1);a=p.parse_args(); started=time.time(); rows=[]
    for split in ('train','geometry_ood'):
        for model_seed in (11,22): rows += run(0,split,model_seed,a.n,correction_scale=a.scale)
    out=ROOT/'development_eval_cup.json'; out.write_text(json.dumps({'status':'development_only','rows':rows,'seconds':time.time()-started},indent=2)+'\n'); print('SUMMARY',json.dumps({'base':sum(r['base'] for r in rows),'gradient':sum(r['gradient'] for r in rows),'n':len(rows)}))
