"""Causal multiview inputs with the same public low-global feedback in every arm."""
from collections import deque
import torch
from prove.fast_adapter.role_dynamics import observed_effect,select_action

class MultiViewState:
    def __init__(self,net,batch,mode='continuous'):
        if mode not in ('fixed','first3','continuous'):raise ValueError(mode)
        if any(p.requires_grad for p in net.parameters()):raise ValueError('runtime weights must be frozen')
        self.net=net;self.mode=mode;self.z=net.z0[None].expand(batch,-1).detach().clone()
        self.obs=[];self.actions=[];self.contexts=[];self.buffer=deque(maxlen=8);self.updates=0
    @torch.no_grad()
    def act(self,ob,base):
        self.obs.append({k:v.detach() for k,v in ob.items()})
        previous=self.actions[-1] if self.actions else torch.zeros_like(base)
        velocity=ob['boxes']-self.obs[-2]['boxes'] if len(self.obs)>1 else torch.zeros_like(ob['boxes'])
        u=self.net.encode(ob['tokens'],ob['boxes'],ob['proprio'],previous,velocity);self.contexts.append(u)
        return select_action(self.net,u,base,self.z)
    @torch.no_grad()
    def observe(self,action,next_ob):
        self.actions.append(action.detach());t=len(self.actions);d={'updated':False,'count':self.updates}
        if t>=4:
            s=t-4;old=self.obs[s];mu,g=self.net.coefficients(self.contexts[s],torch.stack(self.actions[s:t],1))
            y,w=observed_effect(old['target_tokens'],old['target_boxes'],old['proprio'],
                               next_ob['target_tokens'],next_ob['target_boxes'],next_ob['proprio'])
            self.buffer.append((mu,g,y/self.net.effect_sd,w))
            if self.mode=='continuous' or (self.mode=='first3' and self.updates<3):
                mu,g,y,w=[torch.stack(x,1) for x in zip(*self.buffer)];self.z,diag=self.net.update(self.z,mu,g,y,w)
                self.z=self.z.detach();self.updates+=1;d.update(diag,updated=True,count=self.updates)
        return d
