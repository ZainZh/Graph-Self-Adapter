"""P26 actor-only repair, using frozen P25 latent adaptation."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import geometry, domain
from prove.fast_adapter.insertion_teacher import action_labels
from tools.fast_adapter.p25_common import load_adapter as parent_adapter, make_bank, select_feedback, transitions
from tools.fast_adapter.p24_common import load_base, weight_hash

R=Path('evaluations/fast_adapter_p26_actor_repair_v01')
PARENT=Path('evaluations/fast_adapter_p25_feedback_v01')
SEEDS=(11,22)
ARMS=('oldlabel','relabel','refresh')


def adaptation_hash(net):
    h=hashlib.sha256()
    for k,v in sorted(net.state_dict().items()):
        if not k.startswith('actor.'):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_adapter(seed, arm, device='cuda', stage='final'):
    net=parent_adapter(seed,'physical',device)
    if arm!='parent':
        state=torch.load(R/'models'/f'seed{seed}'/stage/f'{arm}.pt',map_location=device,weights_only=True)
        net.load_state_dict(state['state'])
        assert adaptation_hash(net)==state['adaptation_hash']
    return net.eval().requires_grad_(False)


def labels_for(positions, ids):
    configs=[geometry(2201000+int(i)) for i in ids]
    dyn=[domain(2202000+int(i)) for i in ids]
    return action_labels(positions,configs,np.array([x[0] for x in dyn]),np.array([x[1] for x in dyn]))


def prepare_old(net,base):
    bank, paths=make_bank();bank={k:v.cuda() for k,v in bank.items()}
    uv,bb,zz,yy=[],[],[],[]
    for tr in range(6):
        v,p,a=bank['visual'][tr],bank['proprio'][tr],bank['actions'][tr]
        idx=select_feedback(p,a);z=[]
        for first in range(0,384,64):
            sl=slice(first,first+64)
            feedback=transitions(v[sl],p[sl],a[sl],idx[sl])
            z.append(net.feedback_path(feedback)[0][-1].detach())
        prev=torch.cat((torch.zeros_like(a[:,:1]),a[:,:-1]),1)
        with torch.no_grad():
            uv.append(net.encode(v[:,:48],p[:,:48],prev));bb.append(base(v[:,:48],p[:,:48]))
        zz.append(torch.cat(z))
        yy.append(torch.from_numpy(labels_for(p[:,:48].cpu().numpy(),np.arange(384))).cuda())
    return dict(u=torch.stack(uv),base=torch.stack(bb),z=torch.stack(zz),old=bank['expert'],new=torch.stack(yy)),paths


def prepare_fresh(net):
    folder=R/'training_refresh'
    support=torch.load(folder/'SUPPORT.pt',weights_only=False)
    p=torch.tensor(support['positions'],dtype=torch.float32,device='cuda');v=support['visual'].cuda();a=torch.tensor(support['actions'],device='cuda')
    feedback=transitions(v,p,a,select_feedback(p,a));z=net.feedback_path(feedback)[0][-1].detach()
    us,bs,ys=[],[],[]
    for seed in SEEDS:
        raw=torch.load(folder/f'query_seed{seed}_RAW.pt',weights_only=False)
        v=raw['visual'].cuda();p=torch.tensor(raw['positions'],dtype=torch.float32,device='cuda');a=torch.tensor(raw['actions'],device='cuda')
        prev=torch.cat((torch.zeros_like(a[:,:1]),a[:,:-1]),1)
        with torch.no_grad():us.append(net.encode(v[:,:48],p[:,:48],prev))
        bs.append(raw['base'].cuda());ys.append(torch.from_numpy(labels_for(raw['positions'][:,:48],support['training_ids'])).cuda())
    return dict(u=torch.stack(us),base=torch.stack(bs),new=torch.stack(ys),z=z)
