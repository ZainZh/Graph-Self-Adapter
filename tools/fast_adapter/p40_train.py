"""P40 training material and fixed-budget meta-training. No final split access."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.contact_skills import ContactSkill,configuration,STEPS
from prove.fast_adapter.skill_tokens import (SkillControl,SkillAdapter,MotionEffect,PRIORS,KINDS,
    observed_motion,observation_coordinates,prediction,parameters,action_force,LocalVision)

ROOT=Path('evaluations/fast_adapter_p40_skills_v01')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def collect(task):
    out=ROOT/task/'data';out.mkdir(parents=True,exist_ok=False);vision=LocalVision();records=[];features=[]
    forces={'cup':(.5,1.,2.,3.5,5.,7.),'hammer':(.1,.2,.35,.5,.75,1.),'insert':(.3,.6,1.,1.5,2.5,4.)}[task]
    for i in range(64):
        cfg=configuration(task,404000+i//2,'train');rng=np.random.default_rng(405000+i)
        # Two force schedules for every object; all learning labels derive from these records.
        schedule=np.array(forces);schedule=schedule if i%2==0 else schedule[rng.permutation(len(schedule))]
        e=ContactSkill(cfg);controller=SkillControl(task);images=[];masks=[];obs=[];states=[];actions=[];valid=[]
        try:
            for t in range(STEPS+1):
                o,rgb,mask=e.public();ok=bool(o['visible'].all() and np.isfinite(o['positions']).all())
                if not ok:raise RuntimeError(f'Training required role missing: {task} {i} {t}; retain partial source, no truth fallback')
                obs.append(o);states.append(e.state());images.append(rgb);masks.append(mask)
                if t==STEPS:break
                force=float(schedule[min(5,t//12)])
                a=controller.action(o,t,force);actions.append(a.copy());e.step(a)
            f=vision.encode(images,masks)['local'].cpu().half();features.append(f)
            transitions=[observed_motion(task,obs[t],obs[t+1]) for t in range(STEPS)]
            record=dict(config=cfg,obs=obs,states=states,actions=np.array(actions),transitions=transitions)
            records.append(record);np.save(out/f'episode{i:03}.npy',np.array([record],dtype=object),allow_pickle=True)
        finally:e.close()
        if (i+1)%8==0:print(task,'collection',i+1,flush=True)
    torch.save(torch.stack(features),out/'LOCAL_FEATURES.pt')
    js(out/'COLLECT_DONE.json',dict(episodes=64,objects=32,frames=64*(STEPS+1),feature_hash=sha(out/'LOCAL_FEATURES.pt')))

def prepare(task):
    out=ROOT/task/'data';torch.manual_seed(40001)
    f=torch.load(out/'LOCAL_FEATURES.pt',weights_only=True).float().cuda();sample=f[:,::4].flatten(0,1);mean=sample.mean(0)
    _,sv,v=torch.pca_lowrank(sample-mean,q=20,center=False,niter=3);basis=v[:,:16];scale=(sv[:16].square()/(len(sample)-1)).clamp_min(1e-5).sqrt()
    compressed=((f-mean)@basis/scale).cpu();pca=dict(mean=mean.cpu(),basis=basis.cpu(),scale=scale.cpu())
    # VisualTokens global output is unused; matching PCA entry avoids a separate encoder.
    torch.save({'local':pca},out/'PCA.pt');records=[np.load(out/f'episode{i:03}.npy',allow_pickle=True).item() for i in range(64)]
    d={'local':compressed[:,:-1],'df':compressed[:,1:]-compressed[:,:-1]}
    for k in ('motion','actor_dz','known','relation','relation_df'):
        d[k]=torch.tensor(np.array([[x[k] for x in r['transitions']] for r in records]),dtype=torch.float32)
    d['force']=torch.tensor(np.array([r['actions'][:,-1] for r in records]),dtype=torch.float32)
    d['valid']=((d['actor_dz']>.12) if task=='cup' else (d['force']>.03)).float()
    torch.save(d,out/'TRAIN.pt');js(out/'PREPARE.json',dict(train=sha(out/'TRAIN.pt'),pca=sha(out/'PCA.pt'),variance=float(sv[:16].square().sum()/(sample-mean).square().sum()),active_transitions=int(d['valid'].sum())))

def bank(task):return {k:v.cuda() for k,v in torch.load(ROOT/task/'data/TRAIN.pt',weights_only=True).items()}

def effects(task):
    out=ROOT/task/'effects';out.mkdir(parents=True,exist_ok=False);d=bank(task);torch.manual_seed(40002)
    n=MotionEffect().cuda();opt=torch.optim.Adam(n.parameters(),lr=.001)
    for step in range(2000):
        ids=torch.randint(64,(256,),device='cuda');ts=torch.randint(STEPS,(256,),device='cuda')
        move=d['known'][ids,ts].clone();move[:,-1]=d['motion'][ids,ts]
        j=n(d['local'][ids,ts]);pred=torch.einsum('bfc,bc->bf',j,move)
        loss=(pred-d['df'][ids,ts]).square().mean()+.0001*j.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    torch.save(n.cpu().state_dict(),out/'local.pt')
    # Structured identification labels fitted only on training observations.
    z=torch.zeros(32,2,device='cuda',requires_grad=True);opt=torch.optim.Adam([z],lr=.025)
    for _ in range(700):
        e=parameters(task,z).repeat_interleave(2,0)[:,None]
        pred=prediction(task,e,d['force'],d['actor_dz']);scale=d['actor_dz'].clamp_min(.3) if task=='cup' else torch.ones_like(pred)
        error=((pred-d['motion'])/scale).square();loss=(error*d['valid']).sum()/d['valid'].sum().clamp_min(1)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    labels=parameters(task,z).detach().repeat_interleave(2,0);torch.save(labels.cpu(),out/'LABELS.pt')
    js(out/'DONE.json',dict(effect=sha(out/'local.pt'),labels=sha(out/'LABELS.pt'),label_fit_mse=float(loss),source='Training mask/proprio motion only; no mass/friction labels'))

def train(task,kind,seed):
    out=ROOT/task/'models'/kind/f'seed{seed}';out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);d=bank(task)
    head=MotionEffect().cuda();head.load_state_dict(torch.load(ROOT/task/'effects/local.pt',weights_only=True));head.eval().requires_grad_(False)
    with torch.no_grad():d['jacobian']=head(d['local'])
    labels=torch.load(ROOT/task/'effects/LABELS.pt',weights_only=True).cuda();n=SkillAdapter(task,kind).cuda();opt=torch.optim.Adam(n.parameters(),lr=.001 if kind=='history' else .005)
    log=[];start=time.time()
    for step in range(1,601):
        ids=torch.randint(64,(16,),device='cuda');z=n.initial(16);losses=[]
        for t in range(STEPS):
            lo=max(0,t-3);w={k:v[ids,lo:t+1] for k,v in d.items()};z,_=n.update(z,w,meta=True)
            wanted=torch.empty(16,device='cuda').uniform_(.15,1.)
            loss=(action_force(task,n.estimate(z),wanted)-action_force(task,labels[ids],wanted)).square().mean()
            losses.append(loss)
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),5);opt.step()
        if step==1 or step%100==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.time()-start);log.append(row);js(out/'PROGRESS.json',row);print(task,kind,seed,row,flush=True)
    torch.save(n.cpu().state_dict(),out/'model.pt');js(out/'TRAIN.json',dict(task=task,kind=kind,seed=seed,updates=600,log=log,hash=sha(out/'model.pt')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','prepare','effects','train']);p.add_argument('--task',required=True);p.add_argument('--kind',choices=KINDS);p.add_argument('--seed',type=int,default=11);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.mode=='collect':collect(a.task)
    elif a.mode=='prepare':prepare(a.task)
    elif a.mode=='effects':effects(a.task)
    else:train(a.task,a.kind,a.seed)
