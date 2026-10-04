"""Post-seal development diagnostic; never a new controller or task score.

Decode object movement from feature changes with the frozen effect head after
subtracting observed actor movement. Compare equal PCA16 representations against
mask-derived object displacement, with a fixed ridge, and report all transitions.
"""
import json
import torch
from prove.fast_adapter.planar_tokens import FeatureEffect
from tools.fast_adapter.p39_tokens import ROOT,js,sha

def main():
    torch.set_num_threads(2);rows=[];bindings=[]
    for split in ('train','resistance','geometry','joint'):
        p=ROOT/'dev/references/seed0'/split/'base_RUNTIME.pt';r=torch.load(p,weights_only=True,map_location='cpu')
        bindings.append(dict(path=str(p),sha256=sha(p)))
        actor=(r['proprio'][:,1:,:2]-r['proprio'][:,:-1,:2])/.002
        actual=(r['positions'][:,1:,1]-r['positions'][:,:-1,1])/.002
        zero_rmse=float(actual.square().sum(-1).mean().sqrt()*2)
        for kind in ('global','local'):
            hp=ROOT/'effects'/(kind+'.pt');h=FeatureEffect();h.load_state_dict(torch.load(hp,weights_only=True,map_location='cpu'));h.eval().requires_grad_(False)
            f=r[kind];delta=f[:,1:]-f[:,:-1]
            with torch.no_grad():
                j=h(f[:,:-1]);obj=j[...,2:];a=j[...,:2]
                remaining=delta-torch.einsum('btdc,btc->btd',a,actor)
                jt=obj.transpose(-1,-2);matrix=jt@obj+.005*torch.eye(2)
                recovered=torch.linalg.solve(matrix,(jt@remaining[...,None])).squeeze(-1)
                error=recovered-actual
                forward=torch.einsum('btdc,btc->btd',j,torch.cat((actor,actual),-1))
                eig=torch.linalg.eigvalsh(jt@obj)
            rows.append(dict(split=split,token=kind,transitions=actual.shape[0]*actual.shape[1],
                decoded_displacement_rmse_mm=float(error.square().sum(-1).mean().sqrt()*2),
                zero_displacement_rmse_mm=zero_rmse,feature_prediction_mse=float((forward-delta).square().mean()),
                median_weak_axis_sensitivity=float(eig[...,0].median()),ridge=.005))
    js(ROOT/'summary/TOKEN_MOTION_DIAGNOSTIC_DEV.json',dict(scope='Opened development only; no retraining, no simulated rollout and no task-success claim. Fixed ridge0.005, all transitions. Reference motion comes from masks, not evaluator positions.',rows=rows,source_bindings=bindings))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
