"""Descriptive mechanism and cost diagnostics; no candidate changes or new trials."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.skill_tokens_v02 import prediction

ROOT=Path('evaluations/fast_adapter_p40_skills_v01')


def main():
    assert json.loads((ROOT/'FINAL_DONE.json').read_text())['exits']==[0,0,0]
    out={'scope':'Post-seal descriptive analysis, not additional hypothesis tests or model selection', 'tasks':{}}
    for task in ('cup','hammer','insert'):
        rows=[]
        for p in sorted((ROOT/task/'final').rglob('*_RUNTIME.pt')):
            r=torch.load(p,weights_only=True);arm=p.name.removesuffix('_RUNTIME.pt');seal=json.loads(p.with_name(arm+'_SEAL.json').read_text())
            assert hashlib.sha256(p.read_bytes()).hexdigest()==seal['raw']
            result=json.loads(p.with_name(arm+'_RESULT.json').read_text())
            z=r['latent'].float();e=r['parameters'].float();a=r['actions'].float();valid=r['feedback']['valid']>0
            delta=torch.linalg.vector_norm(z[:,1:]-z[:,:-1],dim=-1)
            w=r['feedback'];scale=w['actor_dz'].clamp_min(.3) if task=='cup' else torch.ones_like(w['motion'])
            before=prediction(task,e[:,:-1],w['force'],w['actor_dz'])
            after=prediction(task,e[:,1:],w['force'],w['actor_dz'])
            def masked_rmse(x):
                return float(x[valid].square().mean().sqrt()) if valid.any() else None
            diagnostics=dict(normalized_motion_rmse_before=masked_rmse((before-w['motion'])/scale),
                normalized_motion_rmse_after=masked_rmse((after-w['motion'])/scale))
            if seal['kind'] in ('local','motion','fused','history'):
                j=w['jacobian'];direction=j[...,-1]
                nuisance=torch.einsum('btfc,btc->btf',j[...,:3],w['known'][...,:3])
                inferred=((w['df']-nuisance)*direction).sum(-1)/direction.square().sum(-1).clamp_min(1e-6)
                reconstructed=nuisance+direction*w['motion'][...,None]
                diagnostics.update(visual_projected_motion_rmse=masked_rmse((inferred-w['motion'])/scale),
                    zero_motion_rmse=masked_rmse(w['motion']/scale),
                    feature_prediction_rmse_from_observed_motion=masked_rmse((reconstructed-w['df']).square().mean(-1).sqrt()),
                    note='Least-squares readout of the frozen visual response direction; observed mask/proprio motion reference; descriptive only. Stored actual-trajectory feedback is used, even for donor arms.')
            rows.append(dict(path=str(p),kind=seal['kind'],seed=seal['seed'],split=seal['split'],arm=arm,
                conditions=len(z),successes=result['successes'],valid_first12=valid[:,:12].sum(1).tolist(),valid_all=valid.sum(1).tolist(),
                nonzero_z_updates_first12=(delta[:,:12]>1e-7).sum(1).tolist(),nonzero_z_updates_all=(delta>1e-7).sum(1).tolist(),
                mean_z_update_norm=delta.mean(0).tolist(),mean_parameters=e.mean(0).tolist(),mean_command_force=a[:,:,-1].abs().mean(0).tolist(),
                per_condition_force_mean=a[:,:,-1].abs().mean(1).tolist(),
                same_initial_frame_pairs=seal['split']=='paired',diagnostics=diagnostics))
        out['tasks'][task]=rows
    p=ROOT/'summary/MECHANISM_DIAGNOSTICS.json'
    with p.open('x') as f:json.dump(out,f,indent=2)
    print({k:len(v) for k,v in out['tasks'].items()})


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
