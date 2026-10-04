"""Unify old/new visual numerics BEFORE any P31 task evaluation; no new rollout."""
import argparse
import torch
import numpy as np
from tools.fast_adapter.p31v2_common import *
from tools.fast_adapter.p31v2_data import raw_sources, load_original, qualify_teacher, train_observer
from tools.fast_adapter.p29_common import Visual
from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco


def main(dino):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    out=ROOT/'data';out.mkdir(exist_ok=False)
    qualify_teacher()
    preflight=Path('evaluations/fast_adapter_p31_interaction_stage_v01')
    old=torch.load(preflight/'data/old.pt',weights_only=True)
    own=torch.load(preflight/'data/own.pt',weights_only=True)
    visual=Visual(dino,TASK,torch.load(P29/TASK/'data/PCA.pt',weights_only=True))
    features=[];sources=[];parity=[]
    for role in ('expert','base'):
        for path in raw_sources(role):
            raw,configs=load_original(path);envs=[ObjectEnv(c) for c in configs]
            try:
                for e,seed in zip(envs,raw['reset_seeds']):e.reset(seed)
                frames=[]
                for t in range(201):
                    for j,e in enumerate(envs):
                        for key,values in raw['states'].items():getattr(e.env.data,key)[:]=values[j,t].numpy()
                        mujoco.mj_forward(e.env.model,e.env.data)
                    frames.append(visual.encode(np.stack([e.rgb() for e in envs])).cpu())
                encoded=torch.stack(frames,1);features.append(encoded);sources.append(path)
                if role=='base':
                    offset=int(path.stem.split('_')[-1])
                    error=float((encoded[:,0]-own['visual'][64+offset:72+offset,0]).abs().max())
                    parity.append({'original':str(path),'max_initial_visual_error':error})
                print('REENCODE',path.name,flush=True)
            finally:
                for e in envs:e.close()
    old['visual']=torch.cat(features)
    own['visual'][:64]=old['visual'][:64]
    assert max(x['max_initial_visual_error'] for x in parity)<1e-5,parity
    for label,b in [('old',old),('own',own)]:torch.save(b,out/f'{label}.pt')
    dump(ROOT/'DATA_PARITY.json',{'passed':True,'same_initial_physics':True,'parity':parity,
         'reencoded_frames':128*201,'old_and_new_visual_runtime_matched':True,
         'extra_simulator_rollouts':0,'same_shared_expert_features':True})
    dump(out/'DATA_RECEIPT.json',{'episodes_per_bank':128,'new_physical_episodes':64,
         'bindings':bindings(sources+[preflight/'data/old.pt',preflight/'data/own.pt',ROOT/'VISUAL_FREEZE.json']),
         'outputs':bindings([out/'old.pt',out/'own.pt']), 'all_original_actions_proprio_teacher_near_preserved':True})
    train_observer(old)
    print('PREPARE_V2_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dino',required=True);a=p.parse_args();main(a.dino)
