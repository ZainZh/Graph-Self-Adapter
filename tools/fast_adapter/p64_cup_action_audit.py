"""Train-only action-distillation audit; no adaptation, fitting or physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.p64_cup_policy import CupActionPolicy,PHASES
from prove.fast_adapter.p64_cup import NOMINAL_ROTATION
from prove.fast_adapter.p61_data import sha256


def run(source,checkpoint,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    package=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=CupActionPolicy();model.load_state_dict(package['state_dict']);model.eval().requires_grad_(False)
    centers=[];sizes=[]
    paths=sorted((Path(source)/'data_labels/cup/train').glob('*.npz'))
    if len(paths)!=96:raise ValueError('Only existing 96 training configurations accepted')
    frozen={str(Path(r['label']).resolve()):r['sha256'] for r in package['sources']}
    for path in paths:
        if frozen.get(str(path.resolve()))!=sha256(path):raise ValueError('Training label binding mismatch')
        with np.load(path) as z:centers.append(z['center']);sizes.append(z['size'])
    center=torch.tensor(np.concatenate(centers),dtype=torch.float32)
    size=torch.tensor(np.concatenate(sizes),dtype=torch.float32)
    rotation=torch.tensor(NOMINAL_ROTATION);start=torch.tensor([.07,-.10,.13]).expand_as(center)
    grip=center+(size[:,0:1]-.003)*rotation[:,0]-.004*rotation[:,2]
    results=[]
    with torch.no_grad():
        for i,name in enumerate(PHASES):
            if name=='clearance':
                teacher=start.clone();teacher[:,2]=torch.maximum(start[:,2],torch.full_like(start[:,2],.18))
            elif name=='align':
                teacher=grip+.055*rotation[:,0];teacher[:,2]=torch.maximum(grip[:,2]+.08,torch.full_like(grip[:,2],.18))
            elif name=='descend':teacher=grip+.055*rotation[:,0]
            elif name in ('approach','close'):teacher=grip
            else:teacher=grip+torch.tensor([0.,0.,.065])
            prediction=model(center,size,start,grip,torch.full((len(center),),i,dtype=torch.long))
            error=(prediction-teacher).norm(dim=-1)
            results.append(dict(phase=name,n=len(error),mean_position_error_mm=float(error.mean()*1000),
                p95_position_error_mm=float(torch.quantile(error,.95)*1000),max_position_error_mm=float(error.max()*1000),
                coordinate_rmse_mm=float((prediction-teacher).square().mean().sqrt()*1000)))
    result=dict(schema='prove.p64-cup-training-action-audit.v1',scope='Original training geometry only, nominal yaw0, zero anchor jitter; no physical/heldout outcomes',
        action_checkpoint_sha256=sha256(checkpoint),optimizer_updates=0,physics_steps=0,phases=results)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.source,a.checkpoint,a.output),indent=2))
