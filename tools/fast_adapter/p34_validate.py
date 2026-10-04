"""Fixed offline comparison on reserved training-range objects."""
import numpy as np
import torch
from tools.fast_adapter.p34_common import *
from tools.fast_adapter.p34_train import encode,targets,plans

@torch.no_grad()
def score(b,n):
    count=len(b['actions']);ts=torch.arange(197,device='cuda')[None].expand(count,-1)
    ids=torch.arange(count,device='cuda')[:,None]
    u=encode(n,b,ids,ts);aa=plans(b);yy,w=targets(b);yy=yy/n.effect_sd
    mean,g=n.coefficients(u,aa);out={}
    for mode in ('fixed','first3','continuous'):
        z=n.z0[None].expand(count,-1);errs=[];updates=0
        for t in range(197):
            if t>=4 and (mode=='continuous' or (mode=='first3' and updates<3)):
                lo=max(0,t-11);hi=t-3
                z,_=n.update(z,mean[:,lo:hi],g[:,lo:hi],yy[:,lo:hi],w[:,lo:hi]);updates+=1
            p=n.predict_coeff(mean[:,t],g[:,t],z);errs.append((p-yy[:,t]).square())
        err=torch.stack(errs,1);values={}
        for key,sl in [('all',slice(None)),('features',slice(0,128)),('boxes',slice(128,132)),('relation',slice(132,134))]:
            wt=w[...,sl];valid=wt.sum(-1)>0
            per=(err[...,sl]*wt).sum(-1)/wt.sum(-1).clamp_min(1e-8)
            perobj=(per*valid).sum(-1)/valid.sum(-1).clamp_min(1)
            values[key]={'mean':float(perobj.mean()),'per_episode':perobj.tolist()}
        values['updates_before_last_query']=updates;out[mode]=values
    return out

def main():
    torch.set_num_threads(2);rows=[];selected={};out=ROOT/'validation';out.mkdir(exist_ok=False)
    for task in TASKS:
        b=load_data(task);b={k:v[b['object_ids']>=48] for k,v in b.items() if torch.is_tensor(v)}
        for seed in (11,22):
            for f in FAMILIES:
                n=model(task,seed,f);d=score(b,n);rows.append({'task':task,'seed':seed,'family':f,'metrics':d})
                print('P34_VALIDATE',task,seed,f,{m:round(d[m]['all']['mean'],5) for m in d},flush=True)
        means={f:np.mean([r['metrics']['continuous']['all']['mean'] for r in rows if r['task']==task and r['family']==f]) for f in FAMILIES}
        selected[task]=min(FAMILIES[1:],key=lambda f:means[f]);print('SELECTED',task,selected[task],means,flush=True)
    dump(out/'PREDICTION_RESULTS.json',{'rows':rows,'selection':selected,'all_seeds_retained':True,
         'future_actions_condition_forecast_not_adaptation':True,'readout':'dimensionless training-normalized error, not success'})
    dump(out/'SELECTION_SEAL.json',{'selected':selected,'rule':protocol()['offline_validation']['selection'],
         'bindings':bindings([out/'PREDICTION_RESULTS.json'])})

if __name__=='__main__':main()
