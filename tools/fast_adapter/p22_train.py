"""P22 behavior cloning, followed by matched History/latent-meta training."""
from pathlib import Path
import argparse,copy,json,time
import torch
from prove.fast_adapter.visual_proprio_adapter import VisualBase,ProprioAdapter
from prove.fast_adapter.insert_adapter import support_actions
from tools.fast_adapter.p18_run import save_json,digest
from tools.fast_adapter.p22_data import R

def train(mode,seed):
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cudnn.enabled=False
    out=R/'models'/('base' if mode=='base' else f'seed{seed}');out.mkdir(parents=True,exist_ok=False)
    bank=torch.load(R/'bank/TRAIN.pt',weights_only=True)
    # Explicit allow-list. No slot geometry, scale or hidden dynamics in inputs.
    keys=['support_visual','query_visual','proprio','nominal_label','expert_label','query_action']
    bank={k:bank[k].cuda() for k in keys};n=len(bank['proprio']);began=time.time()
    rng=torch.Generator().manual_seed(seed+22000)
    def sample(batch,k):
        ids=torch.randint(n,(batch,),generator=rng).cuda()
        times=torch.randint(48,(batch,k),generator=rng).cuda()
        bi=ids[:,None].expand(-1,k)
        return ids,bi,times
    log=open(out/'TRAIN.jsonl','w')
    if mode=='base':
        net=VisualBase().cuda();opt=torch.optim.Adam(net.parameters(),lr=.0003)
        for step in range(1,12001):
            _,ids,t=sample(256,1)
            pred=net(bank['query_visual'][ids,t],bank['proprio'][:,6:][ids,t])
            loss=(pred-bank['nominal_label'][ids,t]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step==1 or step%1000==0:
                row={'step':step,'action_mse':loss.item(),'seconds':time.time()-began}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
        torch.save({'state':net.state_dict(),'seed':seed,'updates':12000},out/'model.pt')
    else:
        base=VisualBase().cuda().eval()
        base.load_state_dict(torch.load(R/'models/base/model.pt',weights_only=True)['state']);base.requires_grad_(False)
        net=ProprioAdapter().cuda();opt=torch.optim.Adam(net.parameters(),lr=.0003)
        def phase(name,steps,meta):
            for step in range(1,steps+1):
                ii,ids,t=sample(128,4)
                sv=bank['support_visual'][ii];sp=bank['proprio'][ii,:6];sa=support_actions(len(ii),'cuda')
                z=net.adapt(sv,sp,sa,meta=True)[0] if meta else net.initial(sv,sp,sa)
                zz=z[:,None].expand(-1,4,-1)
                v=bank['query_visual'][ids,t];p=bank['proprio'][:,6:][ids,t]
                with torch.no_grad():ba=base(v,p)
                prediction=net.action(v,p,ba,zz)
                action_loss=((prediction-bank['expert_label'][ids,t])/.35).square().mean()
                transition=net.predict(v,p,bank['query_action'][ids,t],zz)
                dp=(bank['proprio'][:,6:][ids,t+1]-p)/.008
                dv=bank['query_visual'][ids,t+1]-v
                response=(transition[...,:3]-dp).square().mean()
                visual=(transition[...,3:]-dv).square().mean()
                loss=action_loss+.25*response+.025*visual
                if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step==1 or step%500==0:
                    row={'phase':name,'step':step,'action_loss':action_loss.item(),'response_loss':response.item(),
                         'visual_loss':visual.item(),'alpha':net.alpha.item(),'seconds':time.time()-began}
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
        phase('warmup',4000,False)
        warm=copy.deepcopy(net.state_dict());ow=copy.deepcopy(opt.state_dict());rw=rng.get_state()
        torch.save({'state':warm,'updates':4000,'seed':seed},out/'warmup.pt')
        for name,meta in [('history',False),('hybrid',True)]:
            net.load_state_dict(warm);opt.load_state_dict(copy.deepcopy(ow));rng.set_state(rw)
            phase(name,2000,meta)
            torch.save({'state':net.state_dict(),'updates':6000,'seed':seed,'alpha':net.alpha.item()},out/f'{name}.pt')
    log.close()
    save_json(out/'COMPLETE.json',{'mode':mode,'seed':seed,'seconds':time.time()-began,
       'train_sha256':digest(R/'bank/TRAIN.pt'),
       'model_hashes':{p.name:digest(p) for p in out.glob('*.pt')}})
    print('TRAIN_COMPLETE',mode,seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['base','adapter']);p.add_argument('--seed',type=int,default=22011)
    a=p.parse_args();train(a.mode,a.seed)

