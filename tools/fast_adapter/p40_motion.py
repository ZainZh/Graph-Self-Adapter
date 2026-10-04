"""One bounded P40 visual-token revision using existing training frames only."""
import argparse,time
import torch
from tools.fast_adapter.p40_train_v03 import ROOT,js,sha
from prove.fast_adapter.skill_tokens_v02 import MotionEffect,SkillAdapter,action_force
from prove.fast_adapter.contact_skills import STEPS

def prepare(task):
    out=ROOT/task/'motion';out.mkdir(parents=True,exist_ok=False)
    data=ROOT/task/('data_v02' if task=='insert' else 'data');torch.manual_seed(40001)
    f=torch.load(data/'LOCAL_FEATURES.pt',weights_only=True).float().cuda();sample=(f[:,1:]-f[:,:-1])[:,::2].flatten(0,1)
    _,sv,b=torch.pca_lowrank(sample,q=20,center=False,niter=3);basis=b[:,:16];mean=f[:,0].mean(0)
    absolute=(f-mean)@basis;relative=absolute-absolute[:,:1]
    scale=relative.square().mean((0,1)).clamp_min(1e-4).sqrt();token=relative/scale
    torch.save({'local':dict(mean=mean.cpu(),basis=basis.cpu(),scale=scale.cpu())},out/'PCA.pt')
    d=torch.load(data/'TRAIN.pt',weights_only=True);d['local']=token[:,:-1].cpu();d['df']=(token[:,1:]-token[:,:-1]).cpu();torch.save(d,out/'TRAIN.pt')
    variance=float(sv[:16].square().sum()/sample.square().sum());del f,sample,absolute,relative,token
    d={k:v.cuda() for k,v in d.items()};torch.manual_seed(40002);n=MotionEffect().cuda();opt=torch.optim.Adam(n.parameters(),lr=.001)
    for step in range(2000):
        ids=torch.randint(64,(256,),device='cuda');ts=torch.randint(STEPS,(256,),device='cuda');move=d['known'][ids,ts].clone();move[:,-1]=d['motion'][ids,ts]
        j=n(d['local'][ids,ts]);pred=torch.einsum('bfc,bc->bf',j,move);loss=(pred-d['df'][ids,ts]).square().mean()+.0001*j.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    torch.save(n.cpu().state_dict(),out/'local.pt')
    js(out/'PREPARE.json',dict(time=time.time(),effect_updates=2000,pca=sha(out/'PCA.pt'),effect=sha(out/'local.pt'),train=sha(out/'TRAIN.pt'),
        source=sha(data/'LOCAL_FEATURES.pt'),transition_variance_fraction=variance,initial_reference='observed frame0 only',labels_reused=sha(ROOT/task/'effects_v03/LABELS.pt')))

def train(task,seed):
    folder=ROOT/task/'motion';out=folder/f'seed{seed}';out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed)
    d={k:v.cuda() for k,v in torch.load(folder/'TRAIN.pt',weights_only=True).items()};head=MotionEffect().cuda();head.load_state_dict(torch.load(folder/'local.pt',weights_only=True));head.eval().requires_grad_(False)
    with torch.no_grad():d['jacobian']=head(d['local'])
    labels=torch.load(ROOT/task/'effects_v03/LABELS.pt',weights_only=True).cuda();n=SkillAdapter(task,'local').cuda();opt=torch.optim.Adam(n.parameters(),lr=.005);log=[];start=time.time()
    for step in range(1,601):
        ids=torch.randint(64,(16,),device='cuda');z=n.initial(16);losses=[]
        for t in range(STEPS):
            w={k:v[ids,max(0,t-3):t+1] for k,v in d.items()};z,_=n.update(z,w,meta=True)
            wanted=torch.empty(16,device='cuda').uniform_(.15,1.);losses.append((action_force(task,n.estimate(z),wanted)-action_force(task,labels[ids],wanted)).square().mean())
        loss=torch.stack(losses).mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),5);opt.step()
        if step==1 or step%100==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.time()-start);log.append(row);js(out/'PROGRESS.json',row);print(task,seed,row,flush=True)
    torch.save(n.cpu().state_dict(),out/'model.pt');js(out/'TRAIN.json',dict(task=task,kind='motion',seed=seed,updates=600,log=log,hash=sha(out/'model.pt')))

def mean_state(task):
    folder=ROOT/task/'motion';d={k:v.cuda() for k,v in torch.load(folder/'TRAIN.pt',weights_only=True).items()};head=MotionEffect().cuda();head.load_state_dict(torch.load(folder/'local.pt',weights_only=True));head.eval().requires_grad_(False)
    with torch.no_grad():d['jacobian']=head(d['local'])
    means={}
    for seed in (11,22):
        n=SkillAdapter(task,'local').cuda();n.load_state_dict(torch.load(folder/f'seed{seed}'/'model.pt',weights_only=True));n.eval().requires_grad_(False);z=n.initial(64)
        for t in range(STEPS):z,_=n.update(z,{k:v[:,max(0,t-3):t+1] for k,v in d.items()})
        means[f'motion_{seed}']=z.detach().mean(0).cpu()
    path=ROOT/task/'MEANS_motion.pt';assert not path.exists();torch.save(means,path)
    js(ROOT/task/'MEANS_motion.json',dict(hash=sha(path),source='TRAIN only, initial-image referenced'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','mean']);p.add_argument('--task',required=True);p.add_argument('--seed',type=int,default=11);a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.mode=='prepare':prepare(a.task)
    elif a.mode=='mean':mean_state(a.task)
    else:train(a.task,a.seed)
