"""P38 physical data, frozen DINO/LiLa feedback, and bounded continuous trials."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.near_contact_push import NearPush, configuration, DT
from prove.fast_adapter.contact_identification import (ContactAdapter, VisualJacobian,
    PRIOR, force, effect, ridge_response)

ROOT = Path('evaluations/fast_adapter_p38_causal_v01/push')
PROBES = (.15, .65, .35)
KINDS = ('motion', 'appearance', 'combined', 'history')
SEEDS = (11, 22)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Vision:
    def __init__(self, compressed=False):
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        from transformers.utils import logging
        logging.disable_progress_bar()
        from prove.fast_adapter.lila_frontend import ThreeFrontends
        self.net = ThreeFrontends('assets/dinov3', microbatch=8, selected='lila')
        self.pca = None
        if compressed:
            self.pca = {k: v.cuda() for k, v in torch.load(ROOT/'data/PCA.pt', weights_only=True).items()}
    def encode(self, frames):
        value = self.net.encode(np.stack(frames), np.zeros((len(frames), 2, 4), np.float32))[:, 0]
        if self.pca is not None:
            value = (value-self.pca['mean'])@self.pca['basis']/self.pca['scale']
        return value


def collect(shard):
    out = ROOT/'data'/f'shard{shard}'; out.mkdir(parents=True, exist_ok=False)
    vision = Vision(); all_records = []; visual = []
    began = time.time()
    for index in range(24):
        sid = 384000+24*shard+index
        cfg = configuration(sid, 'train'); rng = np.random.default_rng(sid+100000)
        env = NearPush(cfg); obs_list=[]; encoders=[]; actions=[]; frames=[]; states=[]
        for t in range(31):
            obs, own, rgb = env.public(); frames.append(rgb)
            obs_list.append(obs['positions']); encoders.append(own); states.append(env.state())
            if t == 30: break
            error = obs['positions'][2]-obs['positions'][1]
            if t < 3: a = PROBES[t]
            else:
                # Training exploration uses only RGB relation and a fixed nominal
                # controller; independent dithering improves excitation coverage.
                e = torch.tensor([PRIOR]); err = torch.tensor([error])
                a = float(force(err, e)[0])
                a = np.clip(a+rng.uniform(-.13,.13)*(t<23), 0., .8)
            actions.append(a); env.step(a)
        with torch.no_grad():
            f = vision.encode(frames).cpu().half()
        visual.append(f)
        all_records.append({'config':cfg,'positions':np.array(obs_list),'encoder':np.array(encoders),
                            'actions':np.array(actions),'states':states})
        env.close()
        if index % 4 == 0: print('collect',shard,index+1,'seconds',time.time()-began,flush=True)
    torch.save(torch.stack(visual), out/'FEATURES.pt')
    np.save(out/'RECORDS.npy',np.array(all_records,dtype=object),allow_pickle=True)
    save_json(out/'DONE.json',{'episodes':24,'frames':24*31,'seconds':time.time()-began,
                              'features_hash':sha(out/'FEATURES.pt'),'records_hash':sha(out/'RECORDS.npy')})


def merge():
    parts=[torch.load(ROOT/'data'/f'shard{s}/FEATURES.pt',weights_only=True).float() for s in range(4)]
    feature=torch.cat(parts).cuda()
    sample=feature[:,::3].flatten(0,1); mean=sample.mean(0); torch.manual_seed(38030)
    _,sv,basis=torch.pca_lowrank(sample-mean,q=20,center=False,niter=3)
    basis=basis[:,:16]; scale=(sv[:16].square()/(len(sample)-1)).clamp_min(1e-5).sqrt()
    projected=(feature-mean)@basis/scale
    torch.save({'mean':mean.cpu(),'basis':basis.cpu(),'scale':scale.cpu()},ROOT/'data/PCA.pt')
    records=[]
    for s in range(4): records.extend(np.load(ROOT/'data'/f'shard{s}/RECORDS.npy',allow_pickle=True).tolist())
    data={k:torch.tensor(np.stack([r[k] for r in records]),dtype=torch.float32) for k in ('positions','encoder','actions')}
    # Parameters are labels for TRAINING outer-loop supervision only.
    data['response_label']=torch.tensor([[20/(r['config']['damping']+.2),
                             -20*(r['config']['friction']+.01)/(r['config']['damping']+.2)] for r in records])
    data['visual']=projected.cpu(); torch.save(data,ROOT/'data/TRAIN.pt')
    save_json(ROOT/'data/MERGE.json',{'episodes':96,'fit':'training frames only;16 whitened LiLa coordinates',
                'pca_hash':sha(ROOT/'data/PCA.pt'),'train_hash':sha(ROOT/'data/TRAIN.pt'),
                'variance_fraction':float(sv[:16].square().sum()/(sample-mean).square().sum())})


def bank():
    b={k:v.cuda() for k,v in torch.load(ROOT/'data/TRAIN.pt',weights_only=True).items()}
    b['motion']=(b['positions'][:,1:,1]-b['positions'][:,:-1,1])/.002
    b['df']=b['visual'][:,1:]-b['visual'][:,:-1]
    return b


def train_jacobian():
    out=ROOT/'jacobian';out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(38040); b=bank(); n=VisualJacobian().cuda(); opt=torch.optim.Adam(n.parameters(),lr=.001)
    log=[]
    for step in range(1,4001):
        ids=torch.randint(96,(256,),device='cuda');ts=torch.randint(30,(256,),device='cuda')
        visual=b['visual'][ids,ts]; motion=b['motion'][ids,ts]
        pred=n(visual)*motion[:,None]
        loss=(pred-b['df'][ids,ts]).square().mean()+.0001*n(visual).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%1000==0: log.append({'step':step,'loss':float(loss.detach())});print(log[-1],flush=True)
    torch.save(n.cpu().state_dict(),out/'model.pt');save_json(out/'TRAIN.json',{'log':log,
      'inputs':'current frozen DINOv3+LiLa PCA16','target':'next visual change, scaled by RGB-measured motion; no simulator geometry supervision','sha256':sha(out/'model.pt')})


def jacobian_model():
    n=VisualJacobian().cuda();n.load_state_dict(torch.load(ROOT/'jacobian/model.pt',weights_only=True))
    return n.eval().requires_grad_(False)


def train(kind,seed):
    out=ROOT/'models'/kind/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed); b=bank(); visual_model=jacobian_model()
    with torch.no_grad(): j=visual_model(b['visual'][:,:-1])
    n=ContactAdapter(kind).cuda();opt=torch.optim.Adam(n.parameters(),lr=.003 if kind!='history' else .0005)
    log=[];began=time.time()
    for step in range(1,1501):
        ids=torch.randint(96,(16,),device='cuda');z=n.initial(16,'cuda'); losses=[]
        for t in range(30):
            lo=max(0,t-3)
            z,_=n.update(z,b['actions'][ids,lo:t+1],b['motion'][ids,lo:t+1],
                 b['df'][ids,lo:t+1],b['visual'][ids,lo:t+1],j[ids,lo:t+1],meta=True)
            # Query relations differ from the support states; expert force labels
            # use training resistance only and never enter runtime adaptation.
            query_error=.002+.032*torch.rand(16,device='cuda')
            predicted=force(query_error,n.estimate(z))
            target=force(query_error,b['response_label'][ids])
            losses.append((predicted-target).square().mean())
        loss=torch.stack(losses).mean()
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),5.);opt.step()
        if step==1 or step%250==0:
            row={'step':step,'loss':float(loss.detach()),'seconds':time.time()-began,
                 'preconditioner':n.preconditioner().detach().cpu().tolist()};log.append(row)
            save_json(out/'PROGRESS.json',row);print(kind,seed,row,flush=True)
    torch.save(n.cpu().state_dict(),out/'model.pt')
    save_json(out/'TRAIN.json',{'kind':kind,'seed':seed,'updates':1500,'continuous_training_steps':30,
                'batch':16,'log':log,'model_hash':sha(out/'model.pt'),'data_hash':sha(ROOT/'data/TRAIN.pt')})


def load(kind,seed):
    n=ContactAdapter(kind).cuda();n.load_state_dict(torch.load(ROOT/'models'/kind/f'seed{seed}/model.pt',weights_only=True))
    return n.eval().requires_grad_(False)


def evaluate(kind,seed,phase,split,arms):
    nitems=8 if phase=='dev' else 32
    code={'native':0,'id':1,'ood':2}[split]
    base_seed=(386000 if phase=='dev' else 396000)+1000*code
    cfgs=[configuration(base_seed+i,'train' if split=='id' else split) for i in range(nitems)]
    n=None if kind=='references' else load(kind,seed)
    jnet=jacobian_model()
    need_visual=kind in ('appearance','combined','history')
    vision=Vision(compressed=True) if need_visual else None
    out=ROOT/phase/kind/f'seed{seed}'/split;out.mkdir(parents=True,exist_ok=False)
    for arm in arms:
        if (out/(arm+'_RESULT.json')).exists():raise RuntimeError('Refusing to overwrite results')
        envs=[NearPush(c) for c in cfgs]
        z=n.initial(nitems,'cuda') if n is not None else None
        records={'positions':[],'encoder':[],'actions':[],'visual':[],'response':[],'z':[],
                 'qpos':[],'qvel':[],'time':[],'rgb_sha256':[]}
        window={'actions':[],'motion':[],'df':[],'visual':[],'j':[]}
        prev_pos=prev_f=prev_a=None
        for t in range(31):
            public=[env.public() for env in envs]
            pos=torch.tensor(np.stack([p[0]['positions'] for p in public]),device='cuda')
            # Fixed-z and references do not need a visual feature model at all.
            active_feature=need_visual and arm!='frozen' and (arm!='first3' or t<=3)
            f=vision.encode([p[2] for p in public]) if active_feature else torch.zeros(nitems,16,device='cuda')
            with torch.no_grad(): j=jnet(f)
            if t:
                motion=(pos[:,1]-prev_pos[:,1])/.002
                vals=(prev_a,motion,f-prev_f,prev_f,prev_j)
                for key,value in zip(window,vals):window[key].append(value)
                w={key:torch.stack(value[-4:],1) for key,value in window.items()}
                if n is not None and arm!='frozen' and (arm!='first3' or t<=3):
                    if arm=='donor':w={k:v.roll(1,0) for k,v in w.items()}
                    z,_=n.update(z,w['actions'],w['motion'],w['df'],w['visual'],w['j'])
                if arm=='least_squares':
                    w={key:torch.stack(value,1) for key,value in window.items()}
                    estimate=ridge_response(w['actions'],w['motion'])
            if arm=='oracle':
                estimate=torch.tensor([[20/(c['damping']+.2),-20*(c['friction']+.01)/(c['damping']+.2)] for c in cfgs],device='cuda')
            elif n is not None:
                with torch.no_grad():estimate=n.estimate(z)
            elif arm=='base' or t==0:
                estimate=torch.tensor(PRIOR,device='cuda').expand(nitems,-1)
            states=[env.state() for env in envs]
            records['positions'].append(pos.cpu());records['visual'].append(f.cpu())
            records['encoder'].append(torch.tensor(np.stack([p[1] for p in public])))
            records['response'].append(estimate.detach().cpu())
            if z is not None:records['z'].append(z.detach().cpu())
            for key in ('qpos','qvel','time'):records[key].append(torch.tensor(np.stack([s[key] for s in states])))
            records['rgb_sha256'].append([hashlib.sha256(p[2].tobytes()).hexdigest() for p in public])
            if t==30:break
            with torch.no_grad():a=force(pos[:,2]-pos[:,1],estimate)
            if t<3:a=torch.full_like(a,PROBES[t])
            records['actions'].append(a.cpu())
            for env,command in zip(envs,a.cpu().tolist()):env.step(command)
            prev_pos,prev_f,prev_a,prev_j=pos,f,a,j
        runtime={k:torch.stack(v,1) for k,v in records.items() if v and k!='rgb_sha256'}
        runtime['rgb_sha256']=records['rgb_sha256']
        raw=out/(arm+'_RUNTIME.pt');torch.save(runtime,raw)
        save_json(out/(arm+'_SEAL.json'),{'time':time.time(),'runtime_sha256':sha(raw),
           'model_hash':None if n is None else sha(ROOT/'models'/kind/f'seed{seed}/model.pt'),
           'inputs':'RGB color-key relation, tool proprioception; frozen visual features when used',
           'oracle':arm=='oracle','shared_probe_steps':3,'total_physical_steps':30})
        # Object truth and outcome are opened only after saving/sealing trajectory.
        rows=[]
        for env,cfg in zip(envs,cfgs):
            state=env.state();error=abs(state['object_x']-env.goal)
            rows.append({'seed':cfg['seed'],'error':float(error),'speed':float(state['qvel'][1]),
                         'success':bool(error<.0015 and abs(state['qvel'][1])<.005)})
            env.close()
        save_json(out/(arm+'_EVALUATOR.json'),{'configs':cfgs,'rows':rows})
        save_json(out/(arm+'_RESULT.json'),{'kind':kind,'seed':seed,'phase':phase,'split':split,
          'arm':arm,'n':nitems,'successes':sum(r['success'] for r in rows),
          'rate':sum(r['success'] for r in rows)/nitems,'rows':rows})
        print(kind,seed,phase,split,arm,sum(r['success'] for r in rows),'/',nitems,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','merge','jacobian','train','eval'])
    p.add_argument('--shard',type=int);p.add_argument('--kind',choices=KINDS+('references',));p.add_argument('--seed',type=int,default=11)
    p.add_argument('--phase',choices=['dev','final'],default='dev');p.add_argument('--split',choices=['native','id','ood'])
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.mode=='collect':collect(a.shard)
    elif a.mode=='merge':merge()
    elif a.mode=='jacobian':train_jacobian()
    elif a.mode=='train':train(a.kind,a.seed)
    else:
        arms=('base','least_squares','oracle') if a.kind=='references' else ('frozen','continuous','first3','donor')
        for split in ([a.split] if a.split else ['native','id','ood']):evaluate(a.kind,a.seed,a.phase,split,arms)
