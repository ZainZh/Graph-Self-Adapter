"""Re-render old development trajectories with three additional visual streams."""
import argparse,time
import numpy as np
import torch
from tools.fast_adapter.p35_common import *
from tools.fast_adapter.p35_visual import NativeVisual
from tools.fast_adapter.p35_camera import CameraRig
from tools.fast_adapter.p34_branches import raw_source
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
from prove.fast_adapter.held_hammer_env import HeldHammerEnv
from prove.fast_adapter.role_dynamics import observed_effect

def prepare_stats(task,view):
    b=load_data(task,view,'cpu');fit=b['object_ids']<24;t=b['tokens'][fit];mask=t[...,-1:]
    dims=(0,1,3,4);den=mask.sum(dims).clamp_min(1);mean=(t[...,:50]*mask).sum(dims)/den
    sd=(((t[...,:50]-mean[None,None,:,None,None,:]).square()*mask).sum(dims)/den).sqrt().clamp_min(.02)
    tt=b['target_tokens'];bb=b['target_boxes'];p=b['proprio']
    y,w=observed_effect(tt[:,:-4],bb[:,:-4],p[:,:-4],tt[:,4:],bb[:,4:],p[:,4:]);y=y[fit];w=w[fit];ok=(w>0).float()
    esd=((y.square()*ok).sum((0,1))/ok.sum((0,1)).clamp_min(1)).sqrt().clamp_min(.001)
    motion=y[...,128:130];visible=ok[...,128:130]
    direction=(motion*visible).sum((0,1));direction/=direction.norm().clamp_min(1e-8)
    ps=(((motion@direction).square()*visible[...,0]).sum()/visible[...,0].sum().clamp_min(1)).sqrt().clamp_min(.0001)
    torch.save({'token_mean':mean,'token_sd':sd,'effect_sd':esd,'direction':direction,'progress_sd':ps},stats_path(task,view))

def main(task):
    torch.set_num_threads(2);out=ROOT/task/'data';out.mkdir(parents=True,exist_ok=True)
    if (out/'RECEIPT.json').exists():raise FileExistsError('data already sealed')
    mount=json.loads((ROOT/'preflight/WRIST_MOUNT.json').read_text());visual=NativeVisual()
    streams={'global448':(448,'global'),'wrist224':(224,'wrist'),'wrist448':(448,'wrist')}
    parts={k:[] for k in streams};sources=[];seconds={k:0. for k in streams};began=time.time()
    for role in ('expert','base'):
        for offset in (0,8,16,48):
            raw,cfg,j,path=raw_source(task,role,offset)
            mp=path.with_suffix('.json') if task=='hammer-block' else path.with_name(path.stem+'_EVALUATOR.json')
            configs=json.loads(mp.read_text())['configs'];cls=ObjectEnv if task==TASKS[0] else HeldHammerEnv
            envs=[cls(c) for c in configs];rigs=[]
            try:
                for e,s in zip(envs,raw['reset_seeds']):e.reset(s);rigs.append(CameraRig(e,mount))
                frames={k:[] for k in streams}
                for t in range(201):
                    for j,e in enumerate(envs):
                        for k,v in raw['states'].items():getattr(e.env.data,k)[:]=v[j,t].numpy()
                        mujoco.mj_forward(e.env.model,e.env.data)
                    for key,(size,view) in streams.items():
                        start=time.time();obs=[rig.render(size,view) for rig in rigs]
                        v=visual.encode_view(np.stack([x[0] for x in obs]),raw['proprio'][:,t].numpy(),[x[1] for x in obs],task)
                        frames[key].append({k:x.cpu() for k,x in v.items()});seconds[key]+=time.time()-start
                for key in streams:
                    b={k:torch.stack([f[k] for f in frames[key]],1) for k in frames[key][0]}
                    dest=out/f'{key}_{role}_{offset:03d}.pt';torch.save(b,dest);parts[key].append(b)
                sources+=bindings([path,mp]);print('P35_DATA',task,role,offset,'seconds',round(time.time()-began,1),flush=True)
            finally:
                for rig in rigs:rig.close()
                for e in envs:e.close()
    for key in streams:torch.save({k:torch.cat([b[k] for b in parts[key]]) for k in parts[key][0]},out/f'{key}.pt')
    for view in VIEWS:prepare_stats(task,view)
    dump(out/'RECEIPT.json',{'episodes':64,'training_objects':24,'validation_objects':8,'frames':64*201,
        'no_new_demonstrations':True,'native_global_and_wrist_rendering':True,'seconds':time.time()-began,
        'render_and_encode_seconds_by_stream':seconds,'source_bindings':sources,
        'output_bindings':bindings([out/f'{k}.pt' for k in streams]+[stats_path(task,v) for v in VIEWS]),
        'code_bindings':bindings([Path(__file__),Path('tools/fast_adapter/p35_visual.py'),Path('tools/fast_adapter/p35_camera.py'),ROOT/'PROTOCOL.json'])})
    print('P35_DATA_COMPLETE',task,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);a=p.parse_args();main(a.task)
