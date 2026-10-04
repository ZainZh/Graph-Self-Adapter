"""Paired early/middle action choices, frozen before shared physical outcomes."""
import argparse,time
import numpy as np
import torch
from tools.fast_adapter.p35_common import *
from tools.fast_adapter.p35_train import encode,targets,plans
from tools.fast_adapter.p34_branches import raw_source,physical_value
from prove.fast_adapter.role_dynamics import action_candidates,select_action

@torch.no_grad()
def predict():
    torch.set_num_threads(2);out=ROOT/'branches';out.mkdir(exist_ok=False);rows=[];snapshots=[]
    times=(20,40,60)
    for task in TASKS:
        common=load_data(task,'g224');ids=torch.where(common['object_ids']>=48)[0]
        base=base_model(task);base_actions={}
        for t in times:
            ba=base(common['visual'][ids,t],common['proprio'][ids,t],common['actions'][ids,t-1])
            if task=='hammer-block':ba[:,3]=.9
            base_actions[t]=ba;ca=action_candidates(ba,task)
            for j,idx in enumerate(ids.tolist()):
                snapshots.append({'task':task,'episode_index':idx,'object_id':int(common['object_ids'][idx]),
                    'role':'base' if int(common['role'][idx]) else 'expert','t':t,'candidates':ca[j].cpu()})
        for view in VIEWS:
            b=load_data(task,view);y,w=targets(b);aa=plans(b)
            ts=torch.arange(197,device='cuda')[None].expand(16,-1)
            for seed in (11,22):
                for f in FAMILIES:
                    n=model(task,view,seed,f);u=encode(n,b,ids[:,None],ts);mu,g=n.coefficients(u,aa[ids]);yy=y[ids]/n.effect_sd;ww=w[ids]
                    for mode in ('fixed','first3','continuous'):
                        z=n.z0[None].expand(16,-1).clone();updates=0;zs=[];scores=[];choices=[];counts=[]
                        for t in range(61):
                            if t>=4 and (mode=='continuous' or (mode=='first3' and updates<3)):
                                lo=max(0,t-11);hi=t-3;z,_=n.update(z,mu[:,lo:hi],g[:,lo:hi],yy[:,lo:hi],ww[:,lo:hi]);updates+=1
                            if t in times:
                                a,c,s=select_action(n,u[:,t],base_actions[t],z)
                                zs.append(z.cpu().clone());scores.append(s.cpu());choices.append(c.cpu());counts += [updates]*16
                        rows.append({'task':task,'view':view,'seed':seed,'family':f,'mode':mode,
                            'episode_indices':ids.cpu().repeat(3),'times':torch.tensor(times).repeat_interleave(16),
                            'z':torch.cat(zs),'scores':torch.cat(scores),'choice':torch.cat(choices),'updates':counts})
                    print('P35_BRANCH_PREDICT',task,view,seed,f,flush=True)
    torch.save({'snapshots':snapshots,'rows':rows},out/'PREDICTIONS.pt')
    sources=[out/'PREDICTIONS.pt',ROOT/'PROTOCOL.json',Path(__file__),Path('tools/fast_adapter/p35_train.py')]
    sources += [ROOT/t/'models'/v/f'seed{s}'/f'{f}.pt' for t in TASKS for v in VIEWS for s in (11,22) for f in FAMILIES]
    dump(out/'PREDICTION_SEAL.json',{'before_branch_outcomes':True,'snapshot_times':list(times),'snapshots':96,'branches':864,
        'no_future_feedback':True,'physical_grade':'same held-lift or slider progress as P34','bindings':bindings(sources)})

def run():
    from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    torch.set_num_threads(2);out=ROOT/'branches';seal=json.loads((out/'PREDICTION_SEAL.json').read_text())
    for x in seal['bindings']:assert digest(x['path'])==x['sha256'],x['path']
    if (out/'OUTCOMES.pt').exists():raise FileExistsError('outcomes sealed')
    pred=torch.load(out/'PREDICTIONS.pt',weights_only=True);outcomes=[];start_time=time.time()
    for k,snap in enumerate(pred['snapshots']):
        task=snap['task'];raw,cfg,j,path=raw_source(task,snap['role'],snap['object_id']);t=snap['t']
        e=(ObjectEnv if task==TASKS[0] else HeldHammerEnv)(cfg);traces=[];progress=[];frames=[];starts=[]
        try:
            for a in snap['candidates'].numpy():
                e.reset(raw['reset_seeds'][j])
                for key,v in raw['states'].items():getattr(e.env.data,key)[:]=v[j,t].numpy()
                mujoco.mj_forward(e.env.model,e.env.data);v0=physical_value(e,task)
                states=[e.state()];rgb=[e.rgb()];values=[v0]
                for _ in range(4):e.step(a);states.append(e.state());rgb.append(e.rgb());values.append(physical_value(e,task))
                progress.append(values[-1]-v0);starts.append(v0);frames.append(np.stack(rgb))
                traces.append({'states':{key:torch.tensor(np.stack([s[key] for s in states])) for key in states[0]},'values':values})
        finally:e.close()
        assert np.ptp(starts)<1e-8
        outcomes.append({**{key:v for key,v in snap.items() if key!='candidates'},'progress':progress,'start_values':starts,
            'traces':traces,'rgb':torch.tensor(np.stack(frames)),'config':cfg,'reset_seed':int(raw['reset_seeds'][j]),'source':str(path)})
        if k%8==7:print('P35_BRANCH_RUN',k+1,'/96','seconds',round(time.time()-start_time,1),flush=True)
    torch.save({'outcomes':outcomes},out/'OUTCOMES.pt');rows=[];base={}
    for task in TASKS:
        oc=[o for o in outcomes if o['task']==task];v=np.array([o['progress'] for o in oc]);best=v.max(-1);bv=v[:,0]
        base[task]={'progress_mm':float(bv.mean()*1000),'regret_mm':float((best-bv).mean()*1000),
                    'informative_snapshots':int((np.ptp(v,axis=1)>.002).sum()),'snapshots':48}
        for r in pred['rows']:
            if r['task']!=task:continue
            chosen=r['choice'].numpy();actual=v[np.arange(48),chosen];regret=best-actual
            rows.append({key:r[key] for key in ('task','view','seed','family','mode')}|{
                'progress_mm':float(actual.mean()*1000),'regret_mm':float(regret.mean()*1000),
                'per_snapshot_progress_m':actual.tolist(),'per_snapshot_regret_m':regret.tolist(),
                'choices':chosen.tolist(),'beats_base':int((actual>bv+1e-8).sum()),'harms_base':int((actual<bv-1e-8).sum())})
    dump(out/'RESULTS.json',{'rows':rows,'base':base,'branches':864,'scored_action_steps':3456,'seconds':time.time()-start_time,
        'not_full_task_success':True,'bindings':bindings([out/'PREDICTIONS.pt',out/'OUTCOMES.pt',out/'PREDICTION_SEAL.json'])})
    print('P35_BRANCH_COMPLETE',base,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['predict','run']);a=p.parse_args()
    predict() if a.mode=='predict' else run()
