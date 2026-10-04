"""Causal controller: execute one step, then adapt from completed transitions."""
from collections import deque
import torch
from prove.fast_adapter.relational_dynamics import observed_effect,select_action

class DynamicsState:
    def __init__(self,net,batch,mode='continuous'):
        if mode not in ('fixed','first3','continuous'):raise ValueError(mode)
        if any(p.requires_grad for p in net.parameters()):raise ValueError('Runtime network weights must be frozen')
        self.net=net;self.mode=mode;self.z=net.z0[None].expand(batch,-1).detach().clone()
        self.obs=[];self.actions=[];self.contexts=[];self.buffer=deque(maxlen=8)
        self.updates=0;self.diagnostics=[]

    @torch.no_grad()
    def act(self,ob,base):
        self.obs.append({k:v.detach() for k,v in ob.items()})
        prev=self.actions[-1] if self.actions else torch.zeros_like(base)
        vel=ob['boxes']-self.obs[-2]['boxes'] if len(self.obs)>1 else torch.zeros_like(ob['boxes'])
        u=self.net.encode(ob['tokens'],ob['boxes'],ob['proprio'],prev,vel)
        self.contexts.append(u)
        a,idx,scores=select_action(self.net,u,base,self.z)
        return a,idx,scores

    @torch.no_grad()
    def observe(self,actual_action,next_ob):
        self.actions.append(actual_action.detach());t=len(self.actions)
        d={'updated':False,'source_start':None,'source_end':t,'count':self.updates,
           'delta_norm':torch.zeros(len(self.z),device=self.z.device)}
        if t>=4:
            s=t-4;old=self.obs[s]
            plan=torch.stack(self.actions[s:t],1)
            mean,gate=self.net.coefficients(self.contexts[s],plan)
            y,w=observed_effect(old['tokens'],old['boxes'],old['proprio'],
                                next_ob['tokens'],next_ob['boxes'],next_ob['proprio'])
            self.buffer.append((mean,gate,y/self.net.effect_sd,w));d['source_start']=s
            if self.mode=='continuous' or (self.mode=='first3' and self.updates<3):
                means,gates,ys,ws=[torch.stack(x,1) for x in zip(*self.buffer)]
                nz,diag=self.net.update(self.z,means,gates,ys,ws)
                self.z=nz.detach();self.updates+=1;d.update(diag,updated=True,count=self.updates)
        self.diagnostics.append(d)
        return d
