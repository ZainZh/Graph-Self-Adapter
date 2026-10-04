"""Correct visibility weighting of supplementary box/motion diagnostics.

Original overall forecast MSE was already masked and is unchanged. The original
unmasked subgroup fields are retained, explicitly superseded by this sidecar.
No model, control, success count or primary comparison changes.
"""
import torch
from tools.fast_adapter.p33_common import *
from tools.fast_adapter.p33_train import encode,targets,plans

@torch.no_grad()
def main():
    torch.set_num_threads(2);rows=[]
    for split in ('train','geometry_ood'):
        raw=torch.load(ROOT/'development'/f'{split}_sNone_base.pt',weights_only=True)
        b={k:raw[k].cuda() for k in ('tokens','boxes','proprio','actions')}
        yy,w=targets(b);aa=plans(b);n=len(aa)
        ids=torch.arange(n,device='cuda')[:,None];ts=torch.arange(197,device='cuda')[None].expand(n,-1)
        for seed in (11,22):
            for family in ('cross','within'):
                net=model(seed,family);mean,g=net.coefficients(encode(net,b,ids,ts),aa);y=yy/net.effect_sd
                result={}
                for mode in ('fixed','first3','continuous'):
                    z=net.z0[None].expand(n,-1);updates=0;errors=[]
                    for t in range(197):
                        if t>=4 and (mode=='continuous' or (mode=='first3' and updates<3)):
                            lo=max(0,t-11);hi=t-3
                            z,_=net.update(z,mean[:,lo:hi],g[:,lo:hi],y[:,lo:hi],w[:,lo:hi]);updates+=1
                        p=net.predict_coeff(mean[:,t],g[:,t],z);errors.append((p-y[:,t]).square())
                    err=torch.stack(errors,1);row={}
                    for name,sl in [('spatial',slice(0,256)),('boxes',slice(256,264)),('target_motion',slice(260,262)),('proprio',slice(264,268))]:
                        ww=w[...,sl];den=ww.sum((1,2));valid=den>0
                        per=(err[...,sl]*ww).sum((1,2))/den.clamp_min(1e-8)
                        row[name]={'mean':float(per[valid].mean()),'per_condition':per.cpu().tolist(),
                                   'valid_transition_fraction':float((ww.sum(-1)>0).float().mean())}
                    result[mode]=row
                rows.append({'split':split,'seed':seed,'family':family,'diagnostic':result})
    p=ROOT/'diagnostics/FORECAST.json'
    dump(ROOT/'diagnostics/FORECAST_MASKED_SUBGROUPS.json',{'rows':rows,'supersedes_only':['box_mse','target_motion_mse'],
        'original':str(p),'original_sha256':digest(p),'original_all_effect_mse_already_masked_unchanged':True,
        'reason':'The original supplementary subgroup means included missing-entity transitions. This sidecar applies the same observation validity mask as training and overall forecast MSE.',
        'posthoc_diagnostic_only':True})
    print('MASKED_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
