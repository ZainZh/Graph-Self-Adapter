"""Seal public-input predictions first; only then execute shared physical branches."""
import argparse,time
import numpy as np
import torch
from tools.fast_adapter.p34_common import *
from tools.fast_adapter.p34_train import encode,targets,plans
from prove.fast_adapter.role_dynamics import action_candidates,select_action,relative

@torch.no_grad()
def predict():
    torch.set_num_threads(2);out=ROOT/'branches';out.mkdir(exist_ok=False);allrows=[];snapshots=[]
    for task in TASKS:
        b=load_data(task);base=base_model(task);ids=torch.where((b['object_ids']>=48)&(b['object_ids']<56))[0]
        snap=[]
        for idx in ids:
            d=relative(b['boxes'][idx,20:141]).norm(dim=-1)
            ok=b['presence'][idx,20:141].bool().all(-1)
            if not bool(ok.any()):raise ValueError('No publicly valid snapshot; report before proceeding')
            t=20+int(d.masked_fill(~ok,float('inf')).argmin());snap.append(t)
        ts=torch.tensor(snap,device='cuda');prev=b['actions'][ids,ts-1]
        ba=base(b['visual'][ids,ts],b['proprio'][ids,ts],prev)
        if task=='hammer-block':ba[:,3]=.9
        ca=action_candidates(ba,task)
        for j,(idx,t) in enumerate(zip(ids.tolist(),snap)):
            snapshots.append({'task':task,'episode_index':idx,'object_id':int(b['object_ids'][idx]),
                'role':'base' if int(b['role'][idx]) else 'expert','t':t,'candidates':ca[j].cpu()})
        yy,ww=targets(b);aa=plans(b);times=torch.arange(197,device='cuda')[None].expand(len(ids),-1)
        for seed in (11,22):
            for f in FAMILIES:
                n=model(task,seed,f);u=encode(n,b,ids[:,None],times);means,gates=n.coefficients(u,aa[ids]);y=yy[ids]/n.effect_sd;w=ww[ids]
                for mode in ('fixed','first3','continuous'):
                    zs=[];counts=[]
                    for j,t in enumerate(snap):
                        z=n.z0[None].clone();count=0
                        for k in range(4,t+1):
                            if mode=='continuous' or (mode=='first3' and count<3):
                                lo=max(0,k-11);hi=k-3
                                z,_=n.update(z,means[j:j+1,lo:hi],gates[j:j+1,lo:hi],y[j:j+1,lo:hi],w[j:j+1,lo:hi]);count+=1
                        zs.append(z[0]);counts.append(count)
                    z=torch.stack(zs);current=u[torch.arange(len(ids),device='cuda'),ts]
                    act,choice,scores=select_action(n,current,ba,z)
                    allrows.append({'task':task,'seed':seed,'family':f,'mode':mode,
                        'episode_indices':ids.cpu(),'times':ts.cpu(),'z':z.cpu(),'updates':counts,
                        'scores':scores.cpu(),'choice':choice.cpu(),'chosen_action':act.cpu()})
                print('BRANCH_PREDICTIONS',task,seed,f,flush=True)
    torch.save({'snapshots':snapshots,'rows':allrows},out/'PREDICTIONS.pt')
    dump(out/'PREDICTION_SEAL.json',{'before_physical_branch_outcomes':True,'snapshots':32,'branches':288,
        'past_only':True,'cup_grade':'held lift value = max(0,cup_z-initial_z) * bilateral_handle_contact; progress=end-start',
        'hammer_grade':'target slider displacement(end)-displacement(start)',
        'gate_averaging':'all16 snapshots per task; informative count is a separate gate',
        'bindings':bindings([out/'PREDICTIONS.pt',Path(__file__),ROOT/'validation/SELECTION_SEAL.json']+
          [ROOT/t/'models'/f'seed{s}'/f'{f}.pt' for t in TASKS for s in (11,22) for f in FAMILIES])})

def raw_source(task,role,obj):
    off=(obj//8)*8
    if task=='hammer-block':
        path=Path('evaluations/fast_adapter_p30_online_v02/hammer-block/data')/f'meta_{role}_{off:03d}.pt';mp=path.with_suffix('.json')
    else:
        path=Path('evaluations/fast_adapter_p29_object_transfer_v01/tasks')/task/'data/meta'/f'{role}_{off:03d}.pt'
        mp=path.with_name(path.stem+'_EVALUATOR.json')
    raw=torch.load(path,weights_only=True);cfg=json.loads(mp.read_text())['configs'][obj%8]
    return raw,cfg,obj%8,path

def physical_value(e,task):
    if task=='hammer-block':return float(e.env.data.joint('target_slide').qpos[0])
    left,right,*_=e.env.contacts()
    return max(0.,float(e.env.data.body('actor').xpos[2])-e.env.start_z)*float(left and right)

def run():
    from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    torch.set_num_threads(2);out=ROOT/'branches'
    seal=json.loads((out/'PREDICTION_SEAL.json').read_text())
    for x in seal['bindings']:assert digest(x['path'])==x['sha256'],x['path']
    if (out/'OUTCOMES.pt').exists():raise FileExistsError('Physical branch results already exist')
    predictions=torch.load(out/'PREDICTIONS.pt',weights_only=True);outcomes=[];began=time.time()
    for snap in predictions['snapshots']:
        task=snap['task'];raw,cfg,j,path=raw_source(task,snap['role'],snap['object_id']);t=snap['t']
        traces=[];progress=[];starts=[];frames=[]
        e=(HeldHammerEnv if task=='hammer-block' else ObjectEnv)(cfg)
        try:
            for a in snap['candidates'].numpy():
                e.reset(raw['reset_seeds'][j])
                for k,v in raw['states'].items():getattr(e.env.data,k)[:]=v[j,t].numpy()
                mujoco.mj_forward(e.env.model,e.env.data);start=physical_value(e,task)
                states=[e.state()];rgb=[e.rgb()];pp=[];vv=[start]
                for _ in range(4):
                    pp.append(e.step(a));states.append(e.state());rgb.append(e.rgb());vv.append(physical_value(e,task))
                progress.append(vv[-1]-start);starts.append(start);frames.append(np.stack(rgb))
                traces.append({'states':{k:torch.tensor(np.stack([s[k] for s in states])) for k in states[0]},
                               'proprio':torch.tensor(np.stack(pp)),'values':vv})
        finally:e.close()
        outcomes.append({**{k:v for k,v in snap.items() if k!='candidates'},'progress':progress,
            'start_values':starts,'traces':traces,'rgb':torch.tensor(np.stack(frames)),
            'source':str(path),'config':cfg,'reset_seed':int(raw['reset_seeds'][j])})
        assert np.ptp(starts)<1e-8
        print('BRANCHES',task,snap['role'],snap['object_id'],'range_m',round(float(np.ptp(progress)),6),flush=True)
    torch.save({'outcomes':outcomes},out/'OUTCOMES.pt')
    selected=json.loads((ROOT/'validation/SELECTION_SEAL.json').read_text())['selected'];rows=[];gates={}
    for task in TASKS:
        oc=[x for x in outcomes if x['task']==task];progress=np.array([x['progress'] for x in oc])
        informative=np.ptp(progress,axis=1)>.002;best=progress.max(-1);base=progress[:,0]
        for r in predictions['rows']:
            if r['task']!=task:continue
            chosen=r['choice'].numpy();actual=progress[np.arange(16),chosen];regret=best-actual
            rows.append({k:r[k] for k in ('task','seed','family','mode')}|{'mean_progress_m':float(actual.mean()),
                'mean_regret_m':float(regret.mean()),'per_snapshot_progress_m':actual.tolist(),
                'per_snapshot_regret_m':regret.tolist(),'beats_base':int((actual>base+1e-8).sum()),
                'harms_base':int((actual<base-1e-8).sum()),'choices':chosen.tolist()})
        checks=[]
        for s in (11,22):
            chosen=next(r for r in rows if r['task']==task and r['seed']==s and r['family']==selected[task] and r['mode']=='continuous')
            flat=next(r for r in rows if r['task']==task and r['seed']==s and r['family']=='flat' and r['mode']=='continuous')
            checks.append({'seed':s,'beats_flat_regret':chosen['mean_regret_m']<flat['mean_regret_m'],
                           'at_least_base_progress':chosen['mean_progress_m']>=float(base.mean())})
        gates[task]={'selected':selected[task],'informative_snapshots':int(informative.sum()),'snapshots':16,
            'base_mean_progress_m':float(base.mean()),'base_mean_regret_m':float((best-base).mean()),'checks':checks,
            'pass':bool(informative.sum()>=5 and all(c['beats_flat_regret'] and c['at_least_base_progress'] for c in checks))}
    dump(out/'RESULTS.json',{'rows':rows,'gates':gates,'seconds':time.time()-began,
        'not_full_task_success':True,'bindings':bindings([out/'PREDICTIONS.pt',out/'OUTCOMES.pt',out/'PREDICTION_SEAL.json'])})
    print('BRANCH_GATE',gates,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['predict','run']);a=p.parse_args()
    predict() if a.mode=='predict' else run()
