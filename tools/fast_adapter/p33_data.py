"""Re-render P30 TRAINING records; no expert labels enter the P33 bank."""
import time
import numpy as np
import torch
from tools.fast_adapter.p33_common import *
from tools.fast_adapter.p33_visual import RelationVisual
from prove.fast_adapter.held_hammer_env import HeldHammerEnv
from prove.fast_adapter.object_interaction_env import mujoco

def main():
    torch.set_num_threads(2);out=ROOT/'data';out.mkdir(exist_ok=False)
    vis=RelationVisual();parts=[];receipts=[];began=time.time();parity=[]
    for role in ('expert','base'):
        for offset in range(0,64,8):
            path=P30/'data'/f'meta_{role}_{offset:03d}.pt'
            raw=torch.load(path,weights_only=True);meta=json.loads(path.with_suffix('.json').read_text())
            envs=[HeldHammerEnv(c) for c in meta['configs']]
            try:
                for e,s in zip(envs,raw['reset_seeds']):e.reset(s)
                frames=[]
                for t in range(201):
                    for j,e in enumerate(envs):
                        for k in raw['states']:getattr(e.env.data,k)[:]=raw['states'][k][j,t].numpy()
                        mujoco.mj_forward(e.env.model,e.env.data)
                    v=vis.encode(np.stack([e.rgb() for e in envs]))
                    frames.append({k:x.cpu() for k,x in v.items()})
                b={k:torch.stack([f[k] for f in frames],1) for k in frames[0]}
                b.update(actions=raw['actions'].float(),proprio=raw['proprio'].float(),
                         object_ids=torch.arange(offset,offset+8),role=torch.full((8,),role=='base',dtype=torch.long))
                parity.append(float((b['visual']-raw['visual'].float()).abs().max()))
                dest=out/f'{role}_{offset:03d}.pt';torch.save(b,dest);parts.append(b)
                receipts+=bindings([path,path.with_suffix('.json'),dest])
                print('DATA',role,offset,'seconds',round(time.time()-began,1),flush=True)
            finally:
                for e in envs:e.close()
    b={k:torch.cat([x[k] for x in parts]) for k in parts[0]}
    torch.save(b,out/'BANK.pt')
    train=b['object_ids']<48;t=b['tokens'][train];mask=t[...,-1:]
    mean=(t[...,:50]*mask).sum((0,1,2,3))/mask.sum()
    sd=(((t[...,:50]-mean).square()*mask).sum((0,1,2,3))/mask.sum()).sqrt().clamp_min(.02)
    # Targets are fixed observed features, never a trainable target encoder.
    def effects(x):
        return torch.cat((x['tokens'][..., :8].flatten(-3),x['boxes'].flatten(-2),x['proprio']),-1)
    e=effects(b);diff=(e[:,H:]-e[:,:-H])[train]
    esd=diff.flatten(0,1).square().mean(0).sqrt().clamp_min(.001)
    motion=diff[...,256+4:256+6].flatten(0,1)
    direction=motion.sum(0);direction=direction/direction.norm().clamp_min(1e-8)
    progress_sd=(motion@direction).square().mean().sqrt().clamp_min(.0001)
    stats={'token_mean':mean,'token_sd':sd,'effect_sd':esd,'direction':direction,'progress_sd':progress_sd,
           'training_ids':torch.arange(48),'validation_ids':torch.arange(48,64)}
    torch.save(stats,out/'STATS.pt')
    dump(out/'RECEIPT.json',{'episodes':128,'training_objects':48,'validation_objects':16,
         'frames':25728,'new_task_rollouts':0,'seconds':time.time()-began,
         'no_teacher_fields':True,'parent_feature_max_abs_diff':max(parity),
         'direction_from_training_RGB_motion':direction.tolist(),'progress_sd':float(progress_sd),
         'bindings':receipts+bindings([out/'BANK.pt',out/'STATS.pt'])})
    print('DATA_COMPLETE',time.time()-began,flush=True)

if __name__=='__main__':main()
