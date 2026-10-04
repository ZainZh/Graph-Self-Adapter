"""Root-frozen paired P64 cup candidate validation; no implicit launch.

Each CLI invocation executes exactly one cell after verifying the source,
checkpoints, training cache and root's separate physical review.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch

from prove.fast_adapter.p64_cup import configuration,public_pose
from prove.fast_adapter.p64_cup_runtime import CupRuntime
from prove.fast_adapter.p64_cup_policy import CupActionPolicy,NeuralCupController
from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
from prove.fast_adapter.p60_contour import model_state_sha256
from tools.fast_adapter.p45_common import observe_roles,sha,js
from tools.fast_adapter.p47_evaluate import snapshot,evaluate_sealed

STEPS=240


def validate(path):
    path=Path(path);freeze=json.loads(path.read_text())
    review=json.loads((path.parent/'ROOT_PHYSICAL_REVIEW.json').read_text())
    if (freeze.get('schema')!='prove.p64-cup-physical-freeze.v1' or freeze.get('status')!='frozen' or freeze.get('steps')!=STEPS
            or review.get('approved') is not True or review.get('reviewer')!='root'
            or review.get('freeze_sha256')!=sha(path)):
        raise ValueError('Exact root physical freeze/review required')
    expected={(s,a) for s in ('id','size','position') for a in ('fixed','translation_size5')}
    cells=freeze['cells']
    if len(cells)!=24 or set((c['split'],c['arm']) for c in cells)!=expected:
        raise ValueError('Expected three strata x four seeds x two matched arms')
    for split in ('id','size','position'):
        seed_sets=[{c['seed'] for c in cells if c['split']==split and c['arm']==arm} for arm in ('fixed','translation_size5')]
        if len(seed_sets[0])!=4 or seed_sets[0]!=seed_sets[1]:raise ValueError('Unmatched cohort')
    resolved={str(Path(name).resolve()):digest for name,digest in freeze['files'].items()}
    required=[__file__,'prove/fast_adapter/p64_cup.py','prove/fast_adapter/p64_cup_runtime.py','prove/fast_adapter/p64_cup_policy.py',
        'prove/fast_adapter/p64_common.py',freeze['encoder'],freeze['action_policy']]
    if any(str(Path(p).resolve()) not in resolved for p in required):raise ValueError('Missing critical frozen file')
    for name,digest in freeze['files'].items():
        if sha(name)!=digest:raise ValueError('Frozen file changed: '+name)
    return freeze


def run(freeze_path,index,output,device='cuda'):
    freeze=validate(freeze_path);cell=freeze['cells'][index];out=Path(output)
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    js(out/'PRE_RUN.json',dict(index=index,cell=cell,freeze_sha256=sha(freeze_path),runtime_truth=False,steps=STEPS))
    env=RobotGeometrySkill(configuration('cup',cell['seed'],cell['split']))
    try:
        runtime=CupRuntime(freeze['encoder'],freeze['dino_model'],cell['arm'],device)
        policy=CupActionPolicy().to(device)
        payload=torch.load(freeze['action_policy'],map_location=device,weights_only=True)
        if payload['schema']!='prove.p64-cup-action.v1':raise ValueError('Wrong action policy')
        policy.load_state_dict(payload['state_dict']);policy.eval().requires_grad_(False)
        controller=NeuralCupController(env.start_pos,env.start_R,policy)
        before={name:model_state_sha256(model) for name,model in [('encoder',runtime.encoder),('action',policy)]}
        trace=[];observations=[];images=[]
        for t in range(STEPS):
            observation,rgb,masks=observe_roles(env);public=public_pose(observation)
            if not controller.adaptation_frozen:runtime.update(public,rgb,masks,t)
            estimate=runtime.estimate();p,R,closed,stage=controller.command(t,estimate,env.proprio())
            row=dict(t=t,state=snapshot(env),command_p=p.copy(),command_R=R.copy(),closed=bool(closed),stage=stage,
                estimate=copy.deepcopy(estimate),adaptation=copy.deepcopy(runtime.records[-1]) if runtime.records else None,
                controls_before=env.data.ctrl.copy(),neural_action=copy.deepcopy(controller.neural_records[-1]))
            observations.append(dict(t=t,**public))
            if t%4==0:images.append(dict(t=t,rgb=rgb.copy(),masks_packed=np.packbits(masks,axis=-1)))
            env.step(p,R,closed);row['controls_after']=env.data.ctrl.copy();row['robot_tracking']=env.robot_tracking_diagnostics();trace.append(row)
        terminal=snapshot(env)
        np.save(out/'TRACE.npy',np.asarray(trace,dtype=object),allow_pickle=True)
        np.save(out/'OBS.npy',np.asarray(observations,dtype=object),allow_pickle=True)
        np.save(out/'TERMINAL.npy',terminal,allow_pickle=True)
        np.savez_compressed(out/'PUBLIC_IMAGES.npz',t=np.array([r['t'] for r in images]),rgb=np.stack([r['rgb'] for r in images]),masks_packed=np.stack([r['masks_packed'] for r in images]))
        after={name:model_state_sha256(model) for name,model in [('encoder',runtime.encoder),('action',policy)]}
        js(out/'SEAL.json',dict(files={n:sha(out/n) for n in ('TRACE.npy','OBS.npy','TERMINAL.npy','PRE_RUN.json','PUBLIC_IMAGES.npz')},weights_before=before,weights_after=after))
        result=evaluate_sealed(env,trace,terminal,out)
        result.update(index=index,cell=cell,weights_unchanged=before==after,elapsed_seconds=time.monotonic()-start,
            neural_action_steps=sum(bool(r['neural_action']['used']) for r in trace),
            update_events=sum(bool(r['U']) for r in runtime.records),orientation_ood=False,
            final_phase=controller.phase,adaptation_frozen=controller.adaptation_frozen)
        if before!=after:raise RuntimeError('Frozen weights changed')
        js(out/'RESULT.json',result);return result
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();print(json.dumps(run(a.freeze,a.index,a.output,a.device),indent=2))
