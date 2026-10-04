"""Score all models against one identical observed-effect target space."""
import torch
from tools.fast_adapter.p35_common import *
from tools.fast_adapter.p35_train import encode,targets,plans

@torch.no_grad()
def score(b,n):
    count=len(b['actions']);ids=torch.arange(count,device='cuda')[:,None];ts=torch.arange(197,device='cuda')[None].expand(count,-1)
    u=encode(n,b,ids,ts);mu,g=n.coefficients(u,plans(b));y,w=targets(b);y=y/n.effect_sd;out={}
    for mode in ('fixed','first3','continuous'):
        z=n.z0[None].expand(count,-1);errors=[];updates=0
        for t in range(197):
            if t>=4 and (mode=='continuous' or (mode=='first3' and updates<3)):
                lo=max(0,t-11);hi=t-3;z,_=n.update(z,mu[:,lo:hi],g[:,lo:hi],y[:,lo:hi],w[:,lo:hi]);updates+=1
            pred=n.predict_coeff(mu[:,t],g[:,t],z);errors.append((pred-y[:,t]).square())
        err=torch.stack(errors,1);values={}
        for k,sl in [('all',slice(None)),('features',slice(0,128)),('boxes',slice(128,132)),('relation',slice(132,134))]:
            wt=w[...,sl];ok=wt.sum(-1)>0;per=(err[...,sl]*wt).sum(-1)/wt.sum(-1).clamp_min(1e-8)
            per=(per*ok).sum(-1)/ok.sum(-1).clamp_min(1);values[k]={'mean':float(per.mean()),'per_episode':per.tolist()}
        values['updates_before_last_query']=updates;out[mode]=values
    return out

def main():
    torch.set_num_threads(2);out=ROOT/'validation';out.mkdir(exist_ok=False);rows=[]
    for task in TASKS:
        common_target=None
        for view in VIEWS:
            b=load_data(task,view);b={k:v[b['object_ids']>=48] for k,v in b.items()}
            y,w=targets(b)
            if common_target is None:common_target=(y.clone(),w.clone())
            else:assert torch.equal(y,common_target[0]) and torch.equal(w,common_target[1])
            for seed in (11,22):
                for f in FAMILIES:
                    n=model(task,view,seed,f);d=score(b,n);rows.append({'task':task,'view':view,'seed':seed,'family':f,'metrics':d})
                    print('P35_VALIDATE',task,view,seed,f,{m:round(d[m]['all']['mean'],5) for m in d},flush=True)
    dump(out/'PREDICTION_RESULTS.json',{'rows':rows,'identical_feedback_across_visual_arms':True,
          'no_future_effect_used_for_adaptation':True,'conditioning':'recorded future actions are the queried plan, not adaptation labels'})

if __name__=='__main__':main()
