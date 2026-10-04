"""Frozen-budget history versus gradient visual adapter training, P20."""
from __future__ import annotations
import argparse,copy,json,time,random,os
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.insert_adapter import InteractionAdapter,support_actions
from tools.fast_adapter.p18_run import digest,save_json
R=Path('evaluations/fast_adapter_p20_insert3d_v01')

def main(seed,device):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    out=R/'models'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    bank=torch.load(R/'bank/TRAIN.pt',weights_only=False,map_location='cpu')
    # Explicit input allow-list: diagnostic coordinates and hidden dynamics absent.
    bank={k:bank[k].to(device) for k in ['support','query','query_next','base','teacher','query_action']}
    net=InteractionAdapter().to(device);opt=torch.optim.Adam(net.parameters(),lr=.0003)
    rng=torch.Generator(device='cpu').manual_seed(seed+1000)
    began=time.time();log=open(out/'TRAIN.jsonl','w',buffering=1)
    def train(phase,steps,meta):
        for step in range(1,steps+1):
            ids=torch.randint(len(bank['base']),(128,),generator=rng).to(device)
            support=bank['support'][ids];acts=support_actions(len(ids),device)
            z=net.adapt(support,acts,3,meta=True)[0] if meta else net.initial(support,acts)
            zz=z[:,None,:].expand(-1,4,-1)
            prediction=net.action(bank['query'][ids],bank['base'][ids],zz)
            actor=((prediction-bank['teacher'][ids])/.35).square().mean()
            visual_prediction=net.predict(bank['query'][ids],bank['query_action'][ids],zz)
            visual=(visual_prediction-bank['query_next'][ids]).square().mean()
            loss=actor+.25*visual
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step==1 or step%200==0:
                record={'phase':phase,'step':step,'action_mse_normalized':actor.item(),'visual_mse':visual.item(),'alpha':net.alpha.item(),'seconds':time.time()-began}
                log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
    train('shared_warmup',4000,False)
    warm=copy.deepcopy(net.state_dict());warm_opt=copy.deepcopy(opt.state_dict());warm_rng=rng.get_state()
    torch.save({'seed':seed,'state':warm,'updates':4000},out/'warmup.pt')
    for phase,meta in [('history',False),('hybrid',True)]:
        net.load_state_dict(warm);opt.load_state_dict(copy.deepcopy(warm_opt));rng.set_state(warm_rng)
        train(phase,2000,meta)
        torch.save({'seed':seed,'kind':phase,'state':net.state_dict(),'updates':6000,'alpha':net.alpha.item()},out/f'{phase}.pt')
    save_json(out/'COMPLETE.json',{'seed':seed,'seconds':time.time()-began,'history_sha256':digest(out/'history.pt'),'hybrid_sha256':digest(out/'hybrid.pt'),'train_sha256':digest(R/'bank/TRAIN.pt'),'same_warmup_and_batch_sequence':True,'inputs':['support','query','query_next','base','teacher','query_action'],'diagnostic_geometry_access':False})
    print('TRAIN_COMPLETE',seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();main(a.seed,a.device)
