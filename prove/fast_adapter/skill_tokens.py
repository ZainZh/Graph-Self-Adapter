"""P40 structured online identification with three alternative feedback tokens.

Only two environment latent coordinates change online. Vision/effect heads and
the learned gradient preconditioner are frozen at evaluation. Response forms are
declared inductive biases, not a universal neural policy.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prove.fast_adapter.contact_skills import EngineeringControl
from prove.fast_adapter.planar_tokens import VisualTokens

PRIORS={'cup':(.5,2.),'hammer':(.20,6.),'insert':(.7,12.)}
KINDS=('relation','local','fused','history')


class LocalVision(VisualTokens):
    """Same frozen LiLa multilevel fusion and role spatial pooling as P39."""
    def encode(self,images,masks):
        rows=[]
        for start in range(0,len(images),8):
            with torch.inference_mode():
                rgb=torch.as_tensor(np.asarray(images[start:start+8]),device='cuda').permute(0,3,1,2).float()/255
                pix=((rgb-self.net.mean)/self.net.std).to(torch.bfloat16)
                h=self.net.model(pixel_values=pix,output_hidden_states=True,return_dict=True).hidden_states
                fused=self.net.fusion([h[i].float() for i in (-12,-8,-4)])
                side=images[0].shape[0]//16;patch=fused[:,self.net.prefix:].reshape(-1,side*side,1024)
                m=torch.as_tensor(np.asarray(masks[start:start+8]),device='cuda').float();w=F.avg_pool2d(m,16,16).flatten(2)
                w=w/w.sum(-1,keepdim=True).clamp_min(1e-6)
                y,x=torch.meshgrid(torch.linspace(-1,1,side,device='cuda'),torch.linspace(-1,1,side,device='cuda'),indexing='ij')
                spatial=torch.stack((torch.ones_like(x),x,y)).flatten(1)
                local=torch.einsum('brp,kp,bpc->brkc',w,spatial,patch).flatten(1)
                if self.pca:local=self.project(local,'local')
                rows.append(local.detach().clone())
        return {'local':torch.cat(rows)}


def observation_coordinates(task,o):
    pos=o['positions'];p=o['proprio']
    # Functional handle for grasping; driven block for hammer; held peg for insert.
    obj=pos[2] if task=='cup' else pos[1]
    if task=='insert':obj=p[:2]
    relation=np.r_[obj-p[:2],pos[2]-obj,o['sizes'][1],p[2:4]]
    return obj,relation


def observed_motion(task,before,after):
    ob,rb=observation_coordinates(task,before);oa,ra=observation_coordinates(task,after)
    actor=(after['proprio'][:2]-before['proprio'][:2])/.001
    obj=(oa-ob)/.001
    sign=1 if task=='cup' else -1
    # Known actor movement is a nuisance in cup/hammer visual prediction.
    known=np.r_[actor[0],actor[1] if task!='insert' else 0.,obj[0],0.]
    return dict(motion=float(sign*obj[1]),actor_dz=float(actor[1]),known=known.astype(np.float32),
                relation=rb.astype(np.float32)/.02,relation_df=(ra-rb).astype(np.float32)/.02)


def parameters(task,z):
    prior=z.new_tensor([PRIORS[task][0],np.log(PRIORS[task][1])])
    e=z+prior
    return torch.stack((e[...,0].clamp(.02,8.),e[...,1].clamp(np.log(.2),np.log(40.))),-1)


def prediction(task,e,force,actor_dz):
    threshold=e[...,0];gain=e[...,1].exp()
    if task=='cup':
        return actor_dz.clamp_min(0)*torch.sigmoid(gain*(force-threshold))
    motion=gain*F.softplus(4*(force-threshold))/4
    return torch.where(force>.03,motion,torch.zeros_like(motion)).clamp_max(12.)


def action_force(task,e,wanted=None):
    threshold=e[...,0];gain=e[...,1].exp()
    if task=='cup':return (threshold+3./gain).clamp(.3,12.)
    if wanted is None:wanted=torch.full_like(threshold,.6 if task=='hammer' else 1.)
    x=(wanted/gain).clamp_min(1e-5)
    inverse=torch.log(torch.expm1((4*x).clamp_max(25)))/4
    return (threshold+inverse).clamp(.05,5.)


class MotionEffect(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,96),nn.SiLU(),nn.Linear(96,64))
    def forward(self,f):return self.net(f).reshape(*f.shape[:-1],16,4)


class SkillAdapter(nn.Module):
    def __init__(self,task,kind):
        super().__init__();self.task=task;self.kind=kind
        self.log_diag=nn.Parameter(torch.tensor([-.5,-1.]));self.lower=nn.Parameter(torch.zeros(1))
        if kind=='history':
            self.embed=nn.Sequential(nn.Linear(44,64),nn.SiLU());self.gru=nn.GRUCell(64,32);self.decode=nn.Linear(32,2)
            nn.init.zeros_(self.decode.weight);nn.init.zeros_(self.decode.bias)

    def initial(self,n):
        return torch.zeros(n,32 if self.kind=='history' else 2,device=self.log_diag.device,requires_grad=self.kind!='history')

    def estimate(self,z):return parameters(self.task,self.decode(z) if self.kind=='history' else z)

    def update(self,z,w,meta=False):
        if self.kind=='history':
            x=torch.cat((w['local'][:,-1],w['df'][:,-1],w['relation'][:,-1],w['force'][:,-1,None],w['actor_dz'][:,-1,None],w['motion'][:,-1,None],w['valid'][:,-1,None]),-1)
            return self.gru(self.embed(x),z),None
        with torch.enable_grad():
            if not meta:z=z.detach().requires_grad_(True)
            pred=prediction(self.task,self.estimate(z)[:,None],w['force'],w['actor_dz'])
            scale=w['actor_dz'].clamp_min(.3) if self.task=='cup' else torch.ones_like(pred)
            geometric=.5*((pred-w['motion'])/scale).square()
            if self.kind in ('local','fused'):
                motion=w['known'].clone();motion=torch.cat((motion[...,:3],pred[...,None]),-1)
                j=w['jacobian'];p=torch.einsum('btfc,btc->btf',j,motion)
                visual=.5*(p-w['df']).square().sum(-1)/(j[...,-1].square().sum(-1)*scale.square()+.05)
                loss=visual if self.kind=='local' else .5*(visual+geometric)
            else:loss=geometric
            valid=w['valid'];loss=(loss*valid).sum(-1)/valid.sum(-1).clamp_min(1.)
            grad,=torch.autograd.grad(loss.sum(),z,create_graph=meta)
            l=torch.diag(self.log_diag.exp().clamp(.02,3.));l=l+torch.stack((self.lower*0,self.lower*0,self.lower,self.lower*0)).reshape(2,2)
            step=grad@(l@l.T).T;step=step/(1+step.norm(dim=-1,keepdim=True)/1.)
            prior=z.new_tensor([PRIORS[self.task][0],np.log(PRIORS[self.task][1])]);out=parameters(self.task,z-step)-prior
        return (out if meta else out.detach()),loss.detach()


class SkillControl(EngineeringControl):
    """One frozen controller shared by all representations and controls."""
    def __init__(self,task):super().__init__(task,0.)

    def action(self,o,t,force):
        self.force=float(force);a=super().__call__(o,t)
        if self.task=='insert':
            progress=self.initial_proprio[1]-o['proprio'][1]
            a[1]=self.force if progress<.045 and abs(o['proprio'][0]-o['positions'][2,0])<.0015 else 0.
        return a

    def remaining(self,o):
        if self.initial is None:return .018 if self.task=='hammer' else .045
        if self.task=='hammer':return max(0.,.018-(self.initial[1,1]-o['positions'][1,1]))
        if self.task=='insert':return max(0.,.045-(self.initial_proprio[1]-o['proprio'][1]))
        return .035
