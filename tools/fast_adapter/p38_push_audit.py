"""Post-seal action/latent/physics replay, batched without new conditions."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.contact_identification import ContactAdapter, VisualJacobian, PRIOR, force, ridge_response
from prove.fast_adapter.near_contact_push import NearPush
from tools.fast_adapter.p38_push import ROOT, PROBES, sha, save_json

R=ROOT.parent


def audit(path):
    record=json.loads(path.read_text());kind,seed,arm=record['kind'],record['seed'],record['arm']
    prefix=path.name.removesuffix('_RESULT.json');raw_path=path.with_name(prefix+'_RUNTIME.pt')
    seal=json.loads(path.with_name(prefix+'_SEAL.json').read_text())
    assert sha(raw_path)==seal.get('runtime_sha256',seal.get('runtime_hash'))
    raw=torch.load(raw_path,weights_only=True)
    private=json.loads(path.with_name(prefix+'_EVALUATOR.json').read_text());cfgs=private['configs']
    nitems=len(cfgs);contrast='push_contrast' in path.parts
    n=None
    if kind!='references':
        n=ContactAdapter(kind);n.load_state_dict(torch.load(ROOT/'models'/kind/f'seed{seed}/model.pt',weights_only=True))
        n.eval().requires_grad_(False);z=n.initial(nitems,'cpu')
        if arm=='train_mean':z=torch.load(R/'push_contrast/means'/kind/f'seed{seed}.pt',weights_only=True).expand(nitems,-1).clone()
    jnet=VisualJacobian();jnet.load_state_dict(torch.load(ROOT/'jacobian/model.pt',weights_only=True));jnet.eval().requires_grad_(False)
    with torch.no_grad():jac=jnet(raw['visual'])
    dz=da=de=0.
    for t in range(31):
        if t:
            lo=max(0,t-4);a=raw['actions'][:,lo:t]
            y=(raw['positions'][:,lo+1:t+1,1]-raw['positions'][:,lo:t,1])/.002
            df=raw['visual'][:,lo+1:t+1]-raw['visual'][:,lo:t]
            fields=[a,y,df,raw['visual'][:,lo:t],jac[:,lo:t]]
            if n is not None and arm not in ('frozen','train_mean') and (arm!='first3' or t<=3):
                if arm=='donor':
                    fields=[v[torch.arange(nitems)^1] if contrast else v.roll(1,0) for v in fields]
                z,_=n.update(z,*fields)
            if arm=='least_squares':
                lo=max(0,t-4) if contrast else 0
                estimate=ridge_response(raw['actions'][:,lo:t],(raw['positions'][:,lo+1:t+1,1]-raw['positions'][:,lo:t,1])/.002)
        if n is not None:
            with torch.no_grad():estimate=n.estimate(z)
            dz=max(dz,float((z-raw['z'][:,t]).abs().max()))
        elif arm=='oracle':
            estimate=torch.tensor([[20/(c['damping']+.2),-20*(c['friction']+.01)/(c['damping']+.2)] for c in cfgs])
        elif arm=='base' or t==0:
            estimate=torch.tensor(PRIOR).expand(nitems,-1)
        de=max(de,float((estimate-raw['response'][:,t]).abs().max()))
        if t<30:
            a=force(raw['positions'][:,t,2]-raw['positions'][:,t,1],estimate)
            if t<3:a=torch.full_like(a,PROBES[t])
            da=max(da,float((a-raw['actions'][:,t]).abs().max()))
    assert da<1e-4 and dz<3e-4 and de<3e-4,(path,da,dz,de)
    state_error=0.;success=[];success_curves=[];error_curves=[]
    for i,cfg in enumerate(cfgs):
        env=NearPush(cfg,render=False)
        curve=[];errors=[]
        for t in range(31):
            state=env.state()
            error_now=abs(state['object_x']-env.goal)
            errors.append(float(error_now))
            curve.append(bool(error_now<.0015 and abs(state['qvel'][1])<.005))
            for k in ('qpos','qvel'):
                error=np.max(np.abs(state[k]-raw[k][i,t].numpy()));state_error=max(state_error,float(error))
            if t<30:env.step(float(raw['actions'][i,t]))
        err=abs(state['object_x']-env.goal)
        ok=bool(err<.0015 and abs(state['qvel'][1])<.005)
        assert ok==record['rows'][i]['success']
        assert abs(err-record['rows'][i]['error'])<1e-9
        success.append(ok);success_curves.append(curve);error_curves.append(errors);env.close()
    assert state_error<1e-8,(path,state_error)
    rgb_pairs=[]
    if contrast:
        # Separate semantic input invariance from exact raster reproducibility.
        # Preserve bitmap mismatches instead of claiming bitwise RGB identity.
        for i in range(0,nitems,2):
            for key in ('origin','distance','mass','halfwidth'):assert cfgs[i][key]==cfgs[i+1][key]
            assert torch.equal(raw['positions'][i,0],raw['positions'][i+1,0])
            assert torch.equal(raw['encoder'][i,0],raw['encoder'][i+1,0])
            rgb_pairs.append(raw['rgb_sha256'][0][i]==raw['rgb_sha256'][0][i+1])
    return {'file':str(path),'episodes':nitems,'physics_states':nitems*31,'max_action_error':da,
            'max_latent_error':dz,'max_response_error':de,'max_physics_error':state_error,
            'runtime_hash_checked':True,'success_recomputed':True,
            'contrast_initial_geometry_and_proprio_identical':contrast,
            'contrast_initial_RGB_identical':all(rgb_pairs) if contrast else None,
            'contrast_RGB_mismatched_pairs':[i for i,ok in enumerate(rgb_pairs) if not ok],
            'per_episode_success_curve':success_curves,'per_episode_error_curve':error_curves}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='final');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cudnn.enabled=False
    files=sorted((R/'push'/a.phase).rglob('*_RESULT.json'))+sorted((R/'push_contrast'/a.phase).rglob('*_RESULT.json'))
    assert len(files)==140,('incomplete batch',len(files))
    results=[];began=time.time()
    for i,path in enumerate(files[a.shard::a.shards]):
        results.append(audit(path))
        if i%5==0:print(a.shard,i+1,'seconds',time.time()-began,flush=True)
    save_json(R/'summary'/f'PHYSICAL_AUDIT_shard{a.shard}.json',{'phase':a.phase,'rows':results,'seconds':time.time()-began})
