"""Matched actor-only labels and small current-policy data comparison."""
import argparse,copy,json,math,time
import torch
from tools.fast_adapter.p26_common import R,ARMS,load_adapter,load_base,adaptation_hash,prepare_old,prepare_fresh,weight_hash
from tools.fast_adapter.p18_run import save_json,digest


def sample_old(bank,g):
    d=torch.randint(384,(64,),generator=g).cuda();sr=torch.randint(6,(64,),generator=g).cuda()
    qr=(sr+torch.randint(1,6,(64,),generator=g).cuda())%6;t=torch.randint(48,(64,4),generator=g).cuda()
    q,dd=qr[:,None],d[:,None]
    return {k:bank[k][q,dd,t] for k in ('u','base','old','new')} | {'z':bank['z'][sr,d][:,None].expand(-1,4,-1)}


def sample_fresh(bank,g):
    d=torch.randint(32,(16,),generator=g).cuda();q=torch.randint(2,(16,),generator=g).cuda()[:,None]
    t=torch.randint(48,(16,4),generator=g).cuda();dd=d[:,None]
    return {k:bank[k][q,dd,t] for k in ('u','base','new')} | {'z':bank['z'][d][:,None].expand(-1,4,-1)}


def main(seed,stage):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.manual_seed(26000+seed)
    assert json.loads((R/'qualification/RESULT.json').read_text())['passed']
    out=R/'models'/f'seed{seed}'/stage;out.mkdir(parents=True,exist_ok=False)
    base=load_base();base_hash=weight_hash(base)
    reference=load_adapter(seed,'parent');adapt_hash=adaptation_hash(reference)
    bank,source_paths=prepare_old(reference,base)
    if stage=='a':
        names=('oldlabel','relabel');lo,hi=1,4000
        nets={name:copy.deepcopy(reference) for name in names};opts={}
    else:
        names=ARMS;lo,hi=4001,6000
        nets={name:load_adapter(seed,'relabel' if name=='refresh' else name,stage='a') for name in names};opts={}
        fresh=prepare_fresh(reference)
    for name,net in nets.items():
        net.actor.requires_grad_(True)
        opts[name]=torch.optim.Adam(net.actor.parameters(),lr=.0003)
        if stage=='final':
            source='relabel' if name=='refresh' else name
            opts[name].load_state_dict(torch.load(R/'models'/f'seed{seed}'/'a'/f'{source}.pt',weights_only=True,map_location='cuda')['optimizer'])
    g=torch.Generator().manual_seed(26100+seed+(1000 if stage=='final' else 0));began=time.time()
    with (out/'TRAIN.jsonl').open('x') as log:
        for step in range(lo,hi+1):
            batch=sample_old(bank,g)
            fb=sample_fresh(fresh,g) if stage=='final' else None
            row={'step':step,'seconds':time.time()-began}
            for name,net in nets.items():
                b=batch
                if name=='refresh':b={k:torch.cat((batch[k][:48],fb[k]),0) for k in ('u','base','z','new')}
                label=b['old'] if name=='oldlabel' else b['new']
                optimizer=opts[name]
                for group in optimizer.param_groups:group['lr']=.00003+.5*(.0003-.00003)*(1+math.cos(math.pi*(step-1)/6000))
                action=net.action_u(b['u'],b['base'],b['z']);loss=((action-label)/.35).square().mean()
                assert torch.isfinite(loss),(name,step)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.actor.parameters(),5.);assert torch.isfinite(norm)
                optimizer.step()
                if step==lo or step%500==0:row[name]=float(loss.detach())
            if step==lo or step%500==0:log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    assert weight_hash(base)==base_hash
    for name,net in nets.items():
        assert adaptation_hash(net)==adapt_hash,(name,'frozen adaptation changed')
        net.requires_grad_(False)
        torch.save({'state':net.state_dict(),'optimizer':opts[name].state_dict(),'updates':hi,'seed':seed,
            'adaptation_hash':adapt_hash,'parent':'P25 physical','actor_only':True},out/f'{name}.pt')
    save_json(out/'COMPLETE.json',{'seed':seed,'stage':stage,'updates_total':hi,'models':{name:digest(out/f'{name}.pt') for name in names},
        'adaptation_unchanged':True,'adaptation_hash':adapt_hash,'base_unchanged':True,'seconds':time.time()-began,
        'training_sources':[{'path':str(p),'sha256':digest(p)} for p in source_paths]})
    print('TRAIN_COMPLETE',seed,stage,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['a','final'],required=True);a=p.parse_args();main(a.seed,a.stage)
