"""Frozen small cup action policy; code owns stage/gripper transitions only."""
import numpy as np
import torch
from torch import nn
from prove.fast_adapter.robot_geometry_env_p45_v02 import GeometryTaskController
from prove.fast_adapter.p64_cup import NOMINAL_ROTATION

PHASES=('clearance','align','descend','approach','close','transport','hold')


class CupActionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(18,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,3))

    def forward(self, center, size, start, anchor, phase):
        onehot=torch.nn.functional.one_hot(phase,len(PHASES)).to(center.dtype)
        # Relative start/anchor and absolute center permit the fixed table
        # clearance rule; this architecture is not translation-equivariant.
        features=torch.cat((size/.01,(start-center)/.1,(anchor-center)/.1,center/.1,onehot),-1)
        return center+self.net(features)*.1


class NeuralCupController(GeometryTaskController):
    """P45 gates retained; every post-observation position command is neural.

    The analytic superclass proposes stage transitions using public geometry;
    its position output is discarded once the shared 12-step prefix ends.
    Orientation is the same nominal task prior in every arm.
    """
    def __init__(self,start_pos,start_R,policy):
        super().__init__('cup',start_pos,start_R)
        self.policy=policy.eval().requires_grad_(False);self.neural_records=[]

    def command(self,t,estimate,proprio):
        phase=self.phase if self.phase!='observe' else 'clearance'
        p,R,closed,stage=super().command(t,estimate,proprio)
        if t<12 or estimate is None:
            self.neural_records.append(dict(t=int(t),used=False,reason='shared prefix or missing visible target'))
            return p,R,closed,stage
        center,rotation,size=estimate
        anchor=self.anchor[0] if self.anchor is not None else center+rotation[:,0]*(size[0]-.003)-rotation[:,2]*.004
        device=next(self.policy.parameters()).device
        with torch.no_grad():
            values=[torch.as_tensor(x,device=device,dtype=torch.float32)[None] for x in (center,size,self.start_pos,anchor)]
            target=self.policy(*values,torch.tensor([PHASES.index(phase)],device=device))[0].cpu().numpy()
        self.neural_records.append(dict(t=int(t),used=True,phase=phase,target=target.copy(),
            center=np.asarray(center).copy(),size=np.asarray(size).copy(),anchor=np.asarray(anchor).copy()))
        self.last_target=(target.copy(),NOMINAL_ROTATION.copy())
        return target,NOMINAL_ROTATION.copy(),closed,stage


def train_action_policy(centers, sizes, *, steps=1000, seed=11):
    """Training-only teacher distillation on supplied TRAIN geometry.

    No physics, outcomes, development records, or evaluator conditions are read.
    Labels use nominal task orientation, matching the size/position-only test.
    """
    if steps!=1000 or seed!=11:raise ValueError('Frozen action training budget')
    torch.manual_seed(seed)
    centers=torch.as_tensor(centers,dtype=torch.float32).reshape(-1,3)
    sizes=torch.as_tensor(sizes,dtype=torch.float32).reshape(-1,2)
    if len(centers)!=len(sizes) or not torch.isfinite(centers).all() or not torch.isfinite(sizes).all():raise ValueError('Malformed train geometry')
    model=CupActionPolicy();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    start=torch.tensor([.07,-.10,.13]);rotation=torch.tensor(NOMINAL_ROTATION)
    generator=torch.Generator().manual_seed(seed)
    losses=[]
    for step in range(steps):
        ids=torch.randint(len(centers),(256,),generator=generator)
        center=centers[ids];size=sizes[ids];starts=start.expand_as(center)
        grip=center+(size[:,0:1]-.003)*rotation[:,0]-.004*rotation[:,2]
        anchor=grip+(torch.rand((256,3),generator=generator)-.5)*.004
        phase=torch.randint(len(PHASES),(256,),generator=generator)
        target=torch.empty_like(center)
        for i,name in enumerate(PHASES):
            mask=phase==i
            if name=='clearance':
                value=starts.clone();value[:,2]=torch.maximum(starts[:,2],torch.full_like(starts[:,2],.18))
            elif name=='align':
                value=grip+.055*rotation[:,0];value[:,2]=torch.maximum(grip[:,2]+.08,torch.full_like(grip[:,2],.18))
            elif name=='descend':value=grip+.055*rotation[:,0]
            elif name=='approach':value=grip
            elif name=='close':value=anchor
            else:value=anchor+torch.tensor([0.,0.,.065])
            target[mask]=value[mask]
        pred=model(center,size,starts,anchor,phase)
        loss=((pred-target)/.01).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    return model.eval().requires_grad_(False),dict(steps=steps,seed=seed,batch=256,lr=.003,
        final_batch_position_rmse_m=(losses[-1]**.5)*.01,losses=losses)
