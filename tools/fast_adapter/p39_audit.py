"""Offline P39 checks: sealed hashes, latent/action replay and physics replay."""
import argparse
import json
import numpy as np
import torch
from prove.fast_adapter.planar_contact import PlanarPush,controller,projected_force,STEPS
from prove.fast_adapter.planar_tokens import PlanarAdapter,FeatureEffect,PRIOR
from tools.fast_adapter.p39_tokens import ROOT,sha,js,probes
from tools.fast_adapter.p39_evaluate import windows

def audit(phase):
    heads={}
    for k in ('global','local'):
        h=FeatureEffect();h.load_state_dict(torch.load(ROOT/'effects'/(k+'.pt'),weights_only=True,map_location='cpu'));heads[k]=h.eval().requires_grad_(False)
    summary=dict(batches=0,episodes=0,states=0,max_action_error=0.,max_latent_error=0.,max_physics_error=0.,max_label_error=0.,max_object_mask_error=0.,failures=[])
    for file in sorted((ROOT/phase).glob('*/seed*/*/*_RUNTIME.pt')):
        arm=file.name.removesuffix('_RUNTIME.pt');out=file.parent;split=out.name;seed=int(out.parent.name[4:]);kind=out.parent.parent.name
        r=torch.load(file,weights_only=True,map_location='cpu');seal=json.loads((out/(arm+'_SEAL.json')).read_text());assert sha(file)==seal['sha256']
        ev=json.loads((out/(arm+'_EVALUATOR.json')).read_text());configs=ev['configs'];nitems=len(configs)
        if kind!='references':
            n=PlanarAdapter(kind);p=ROOT/'models'/kind/f'seed{seed}'/'model.pt';assert sha(p)==seal['model_hash']
            n.load_state_dict(torch.load(p,weights_only=True,map_location='cpu'));n.eval().requires_grad_(False);z=n.initial(nitems)
            if arm=='mean':z=torch.load(p.with_name('TRAIN_MEAN.pt'),weights_only=True)[None].expand(nitems,-1).clone()
        else:n=None;z=None
        donor=None
        if arm=='donor':
            d=torch.load(ROOT/phase/'references'/'seed0'/split/'base_RUNTIME.pt',weights_only=True,map_location='cpu');donor=windows(d,heads)
        envs=[PlanarPush(c,render=False) for c in configs]
        try:
            for t in range(STEPS+1):
                if t and n is not None and arm in ('continuous','first3','donor') and (arm!='first3' or t<=3):
                    if arm=='donor':
                        ids=(torch.arange(nitems)^1) if split=='paired' else torch.arange(nitems).roll(1);w={k:v[ids,max(0,t-4):t] for k,v in donor.items()}
                    else:
                        short={k:r[k][:,max(0,t-4):t+1] for k in ('positions','proprio','global','local','relation','force')}
                        w=windows(short,heads)
                    nz,_=n.update(z,w);z=torch.where(r['valid'][:,t,None],nz,z)
                if n is not None:
                    ze=float((z-r['z'][:,t]).abs().max());summary['max_latent_error']=max(summary['max_latent_error'],ze)
                    assert ze<.001,(str(file),t,'latent',ze)
                    response=n.response(z).detach().numpy()
                else:response=r['response'][:,t].numpy()
                for i,(e,c) in enumerate(zip(envs,configs)):
                    state=e.state();pe=max(float(np.max(np.abs(state[k]-r[k][i,t].numpy()))) for k in ('qpos','qvel'))
                    summary['max_physics_error']=max(summary['max_physics_error'],pe);assert pe<1e-8,(str(file),t,'physics',pe)
                    if r['valid'][i,t]:summary['max_object_mask_error']=max(summary['max_object_mask_error'],float(np.linalg.norm(r['positions'][i,t,1].numpy()-state['object'])))
                    if t<STEPS:
                        obs={k:r[k][i,t].numpy() for k in ('positions','radii','proprio')}
                        expected=probes(obs,t) if t<3 else controller(obs,response[i])
                        if not r['valid'][i,t]:expected=np.zeros(2)
                        actual=r['actions'][i,t].numpy();ae=float(np.max(np.abs(expected-actual)))
                        summary['max_action_error']=max(summary['max_action_error'],ae);assert ae<.002,(str(file),t,'action',ae)
                        e.step(actual)
                    else:
                        error=float(np.linalg.norm(state['object']-e.goal));speed=float(np.linalg.norm(state['qvel'][2:]))
                        row=ev['rows'][i];le=abs(error-row['error']);summary['max_label_error']=max(summary['max_label_error'],le)
                        assert le<1e-10 and bool(r['valid'][i,t] and error<.0025 and speed<.008)==row['success']
        finally:
            for e in envs:e.close()
        summary['batches']+=1;summary['episodes']+=nitems;summary['states']+=nitems*(STEPS+1)
        if summary['batches']%20==0:print(summary,flush=True)
    expected=(5 if phase=='final' else 4)*(2+5*2*4);assert summary['batches']==expected,(summary['batches'],expected)
    summary['passed']=True;js(ROOT/'summary'/(phase+'_AUDIT.json'),summary);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='final');a=p.parse_args();torch.set_num_threads(2);audit(a.phase)
