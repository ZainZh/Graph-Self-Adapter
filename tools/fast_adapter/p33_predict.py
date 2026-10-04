"""Common-record forecast diagnostic, no physics or outcome-driven training.

Future recorded actions condition a retrospective four-step prediction. The
latent only consumes completed earlier effects. This is not a measurement of
the open-loop execution of the controller's originally planned four actions.
"""
import torch
import numpy as np
from tools.fast_adapter.p33_common import *
from tools.fast_adapter.p33_train import encode,targets,plans

@torch.no_grad()
def score(b,net):
    count=len(b['actions']);times=torch.arange(197,device='cuda')[None].expand(count,-1)
    ids=torch.arange(count,device='cuda')[:,None]
    u=encode(net,b,ids,times);aa=plans(b);yy,w=targets(b);yy=yy/net.effect_sd
    mean,g=net.coefficients(u,aa)
    out={}
    for mode in ('fixed','first3','continuous'):
        z=net.z0[None].expand(count,-1);mse=[];box=[];motion=[];zero=[];updates=0
        for t in range(197):
            if t>=4 and (mode=='continuous' or (mode=='first3' and updates<3)):
                lo=max(0,t-4-7);hi=t-3
                z,_=net.update(z,mean[:,lo:hi],g[:,lo:hi],yy[:,lo:hi],w[:,lo:hi]);updates+=1
            pred=net.predict_coeff(mean[:,t],g[:,t],z)
            mse.append(net.loss(pred,yy[:,t],w[:,t]).cpu())
            box.append((pred[:,256:264]-yy[:,t,256:264]).square().mean(-1).cpu())
            motion.append((pred[:,260:262]-yy[:,t,260:262]).square().mean(-1).cpu())
            zero.append(net.loss(torch.zeros_like(pred),yy[:,t],w[:,t]).cpu())
        def stats(xs):
            a=torch.stack(xs,1);return {'mean':float(a.mean()),'per_condition':a.mean(-1).tolist()}
        out[mode]={'all_effect_mse':stats(mse),'box_mse':stats(box),'target_motion_mse':stats(motion),
                   'zero_change_reference':stats(zero),'updates_before_last_query':updates}
    return out

def main():
    torch.set_num_threads(2);out=ROOT/'diagnostics';out.mkdir(exist_ok=False)
    bank=load_data();take=bank['object_ids']>=48
    val={k:v[take] for k,v in bank.items() if torch.is_tensor(v)}
    datasets={'training_range_validation':val}
    for split in ('train','geometry_ood'):
        raw=torch.load(ROOT/'development'/f'{split}_sNone_base.pt',weights_only=True)
        datasets[split]={k:raw[k].cuda() for k in ('tokens','boxes','proprio','actions')}
    rows=[]
    for seed in (11,22):
        for family in ('cross','within'):
            net=model(seed,family)
            for split,b in datasets.items():
                r={'seed':seed,'family':family,'split':split,'diagnostic':score(b,net)};rows.append(r)
                print('PREDICT',seed,family,split,{k:round(v['all_effect_mse']['mean'],4) for k,v in r['diagnostic'].items()},flush=True)
    dump(out/'FORECAST.json',{'rows':rows,'same_records_across_modes':True,'adaptation_never_reads_future_effects':True,
       'conditioning':'Four future recorded actions are supplied as the queried plan, not as adaptation feedback.',
       'data_scope':'Unused training-range validation objects plus new-development BASE trajectories; no new physical trials.',
       'scope_limit':'Not success causality, not held-out final, and not outcomes of executing the original receding-horizon plan.'})

if __name__=='__main__':main()
