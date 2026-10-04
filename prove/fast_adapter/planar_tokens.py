"""P39 frozen visual tokens and structured online response adaptation.

Global and local tokens have identical PCA16 sizes and learned motion-to-feature
heads. Relative geometry is a declared mask-derived input, not DINO knowledge.
The fused arm preserves a geometry-only control to identify its contribution.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PRIOR=((20/6.2,-20*.075/6.2),(20/6.2,-20*.075/6.2))
KINDS=('global','local','relation','fused','history')


class VisualTokens:
    def __init__(self,pca=None):
        from prove.fast_adapter.lila_frontend import ThreeFrontends
        self.net=ThreeFrontends('assets/dinov3',microbatch=8,selected='lila')
        self.pca=pca

    def encode(self,images,masks):
        global_rows=[];local_rows=[]
        for start in range(0,len(images),8):
            with torch.inference_mode():
                rgb=torch.as_tensor(np.array(images[start:start+8]),device='cuda').permute(0,3,1,2).float()/255
                pix=((rgb-self.net.mean)/self.net.std).to(torch.bfloat16)
                hidden=self.net.model(pixel_values=pix,output_hidden_states=True,return_dict=True).hidden_states
                fused=self.net.fusion([hidden[i].float() for i in (-12,-8,-4)])
                g=self.net.adapter(fused).flatten(1)
                side=images[0].shape[0]//16
                patches=fused[:,self.net.prefix:].reshape(-1,side*side,1024)
                m=torch.as_tensor(np.array(masks[start:start+8]),device='cuda').float()
                weight=F.avg_pool2d(m,16,16).flatten(2)
                weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-6)
                y,x=torch.meshgrid(torch.linspace(-1,1,side,device='cuda'),torch.linspace(-1,1,side,device='cuda'),indexing='ij')
                # Appearance plus spatial feature moments; no geometric truth is read.
                spatial=torch.stack((torch.ones_like(x),x,y)).flatten(1)
                local=torch.einsum('brp,kp,bpc->brkc',weight,spatial,patches).flatten(1)
                if self.pca:
                    g=self.project(g,'global');local=self.project(local,'local')
                global_rows.append(g.detach().clone());local_rows.append(local.detach().clone())
        return dict(global_=torch.cat(global_rows),local=torch.cat(local_rows))

    def project(self,x,kind):
        p=self.pca[kind]
        return (x-p['mean'])@p['basis']/p['scale']


class FeatureEffect(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,64))
    def forward(self,x):return self.net(x).reshape(*x.shape[:-1],16,4)


def estimate(z):
    e=z.reshape(*z.shape[:-1],2,2)+z.new_tensor(PRIOR)
    return torch.stack((e[...,0].clamp(.9,8),e[...,1].clamp(-2,0)),-1)


def response_motion(e,applied):
    return applied.sign()*(e[...,0]*applied.abs()+e[...,1]).clamp_min(0)


def desired_force(error,e):
    norm=error.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    wanted=error*(.09/norm).clamp_max(4.)*20
    force=wanted.sign()*(wanted.abs()-e[...,1])/e[...,0]
    return torch.where(norm<.0018,torch.zeros_like(force),force).clamp(-1.5,1.5)


def relation(pos):
    return torch.cat((pos[...,1,:]-pos[...,0,:],pos[...,2,:]-pos[...,1,:]),-1)/.02


class PlanarAdapter(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.log_diag=nn.Parameter(torch.full((4,),-.4))
        self.lower=nn.Parameter(torch.zeros(6))
        if kind=='history':
            self.embed=nn.Sequential(nn.Linear(44,64),nn.SiLU());self.gru=nn.GRUCell(64,32)
            self.decode=nn.Linear(32,4);nn.init.zeros_(self.decode.weight);nn.init.zeros_(self.decode.bias)

    def initial(self,n):return torch.zeros(n,32 if self.kind=='history' else 4,device=next(self.parameters()).device,requires_grad=self.kind!='history')
    def response(self,z):return estimate(self.decode(z) if self.kind=='history' else z)

    def preconditioner(self):
        l=torch.diag(self.log_diag.exp().clamp(.02,4))
        ix=torch.tril_indices(4,4,-1,device=l.device);l=l.index_put((ix[0],ix[1]),self.lower)
        return l@l.T

    def update(self,z,w,meta=False):
        if self.kind=='history':
            t=torch.cat((w['local'][:,-1],w['local_df'][:,-1],w['relation'][:,-1],w['relation_df'][:,-1],w['force'][:,-1],w['actor_motion'][:,-1]),-1)
            return self.gru(self.embed(t),z),None
        with torch.enable_grad():
            if not meta:z=z.detach().requires_grad_(True)
            motion=response_motion(self.response(z)[:,None],w['force'])
            actor=w['actor_motion']
            predicted_relation=.1*torch.cat((motion-actor,-motion),-1)
            # The analytic derivative has squared Frobenius norm .04.
            geometric=.5*(predicted_relation-w['relation_df']).square().sum(-1)/.04
            if self.kind in ('global','local','fused'):
                k='global' if self.kind=='global' else 'local'
                j=w[k+'_j'];joint=torch.cat((actor,motion),-1)
                pred=torch.einsum('btdc,btc->btd',j,joint)
                visual=.5*(pred-w[k+'_df']).square().sum(-1)/(j[...,2:].square().sum((-1,-2))+.05)
                loss=.5*(geometric+visual) if self.kind=='fused' else visual
            else:loss=geometric
            loss=loss.mean(-1)
            grad,=torch.autograd.grad(loss.sum(),z,create_graph=meta)
            step=grad@self.preconditioner().T
            step=step/(1+step.norm(dim=-1,keepdim=True)/3)
            out=(estimate(z-step)-z.new_tensor(PRIOR)).flatten(-2)
        return (out if meta else out.detach()),loss.detach()
