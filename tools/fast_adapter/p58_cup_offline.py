#!/usr/bin/env python3
"""Offline causal P58 cup diagnostics on the opened 12-condition dev pack."""
import argparse,json
from pathlib import Path
import numpy as np
import torch

from prove.fast_adapter.p58_cup_functional import points_to_pose
from prove.fast_adapter.p58_cup_runtime import load_runtime
from tools.fast_adapter.p58_cup import _load,sha


def angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1))))


def run(pack,checkpoints,output):
    public,labels=_load(pack,'dev');rows=[]
    with np.load(Path(pack)/'dev_public.npz',allow_pickle=False) as source:seeds=source['seeds'].copy()
    truth_pose=points_to_pose(labels['points_world'])
    for rep,path in checkpoints.items():
      for arm in ('fixed','online','direct'):
       runtime=load_runtime(path,arm)
       before={k:v.detach().clone() for k,v in runtime.model.state_dict().items()}
       for condition,seed in enumerate(seeds):
        runtime.reset();trajectory=[]
        for t in range(public['features'].shape[1]):
         # Exercise the frozen causal math using compact public inputs directly.
         r=runtime.readout;n=runtime.model.config.point_count;feature=public['features'][condition,t]
         pc=((feature-r['mean'])/r['scale'])@r['components'].T;pred=np.r_[1.,pc]@r['coef']
         bbox=feature[110:114];center=(bbox[:2]+bbox[2:])*.5;extent=np.maximum(bbox[2:]-bbox[:2],1/448.)
         uv=(center[None]+extent[None]*pred[:2*n].reshape(n,2))*448.
         xyz=pred[2*n:].reshape(n,3)*.01
         from prove.fast_adapter.p58_cup_functional import camera_to_world,ray_system,support_window
         base=camera_to_world(xyz,public['camera_pose'][condition,t]);A,C=ray_system(uv,public['K'][condition,t],public['camera_pose'][condition,t])
         valid=bool(public['mask_valid'][condition,t,2])
         if valid:runtime.history=support_window(runtime.history+[(A,C,np.ones(n,bool))])
         At=torch.tensor(np.stack([x[0] for x in runtime.history]),dtype=torch.float32)[None]
         Ct=torch.tensor(np.stack([x[1] for x in runtime.history]),dtype=torch.float32)[None]
         Vt=torch.tensor(np.stack([x[2] for x in runtime.history]),dtype=torch.bool)[None]
         g=torch.tensor(base,dtype=torch.float32)[None];pre=runtime.z.clone()
         if len(runtime.history)>=3 and arm=='online':runtime.z,_=runtime.model.adapt(g,runtime.z,At,Ct,Vt)
         elif len(runtime.history)>=3 and arm=='direct':runtime.z=runtime.model.direct(g,runtime.z,At,Ct,Vt).detach()
         points=runtime.model.decode(g,runtime.z).detach().numpy()[0]
         R=points_to_pose(points)[:3,:3] if n==3 else runtime.mean_rotation
         trajectory.append(dict(t=t,S=len(runtime.history),center_mm=float(np.linalg.norm(points[0]-labels['points_world'][condition,t,0])*1000),
             angle_deg=angle(R,truth_pose[condition,t,:3,:3]),latent_change=float((runtime.z-pre).norm())))
        rows.append(dict(representation=rep,arm=arm,condition=int(seed),split='id' if condition<6 else 'geometry',
            frames=len(trajectory),updates=sum(x['latent_change']>0 for x in trajectory),
            final_center_mm=trajectory[-1]['center_mm'],final_angle_deg=trajectory[-1]['angle_deg'],trajectory=trajectory))
       assert all(torch.equal(before[k],v) for k,v in runtime.model.state_dict().items())
    summary={}
    for rep in checkpoints:
     for arm in ('fixed','online','direct'):
      for split in ('id','geometry'):
       group=[r for r in rows if (r['representation'],r['arm'],r['split'])==(rep,arm,split)]
       summary[f'{rep}/{arm}/{split}']=dict(n=len(group),center_mm=float(np.mean([r['final_center_mm'] for r in group])),
           angle_deg=float(np.mean([r['final_angle_deg'] for r in group])),updates=sum(r['updates'] for r in group))
    value=dict(schema='prove.p58-cup-opened-dev-geometry.v1',pack_manifest_sha256=sha(Path(pack)/'MANIFEST.json'),
        checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in checkpoints.items()},summary=summary,rows=rows,
        scope='opened dev, zero physics; no model selection or tuning')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return value


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pack',required=True);p.add_argument('--position-only',required=True);p.add_argument('--functional-frame',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 run(a.pack,{'position_only':a.position_only,'functional_frame':a.functional_frame},a.output)
