"""Re-encode only previously collected cup TRAINING frames; reuse frozen hammer bank."""
import time
import numpy as np
import torch
from PIL import Image,ImageDraw
from tools.fast_adapter.p34_common import *
from tools.fast_adapter.p34_visual import RoleVisual,calibration,gripper_box
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
from prove.fast_adapter.role_dynamics import observed_effect

def prepare_stats(task):
    b=load_data(task,'cpu');train=b['object_ids']<48;t=b['tokens'][train];mask=t[...,-1:]
    mean=(t[...,:50]*mask).sum((0,1,2,3))/mask.sum()
    sd=(((t[...,:50]-mean).square()*mask).sum((0,1,2,3))/mask.sum()).sqrt().clamp_min(.02)
    y,w=observed_effect(b['tokens'][:,:-4],b['boxes'][:,:-4],b['proprio'][:,:-4],
                        b['tokens'][:,4:],b['boxes'][:,4:],b['proprio'][:,4:])
    y=y[train];w=w[train];available=(w>0).float()
    esd=((y.square()*available).sum((0,1))/available.sum((0,1)).clamp_min(1)).sqrt().clamp_min(.001)
    motion=y[...,128:130];ok=available[...,128:130]
    direction=(motion*ok).sum((0,1));direction=direction/direction.norm().clamp_min(1e-8)
    progress_sd=((motion@direction).square()*ok[...,0]).sum()/ok[...,0].sum().clamp_min(1)
    stats={'token_mean':mean,'token_sd':sd,'effect_sd':esd,'direction':direction,
           'progress_sd':progress_sd.sqrt().clamp_min(.0001)}
    stats_path(task).parent.mkdir(parents=True,exist_ok=True);torch.save(stats,stats_path(task))
    dump(stats_path(task).with_suffix('.json'),{'fit_object_ids':list(range(48)),
         'direction_from_training_RGB_motion':direction.tolist(),'object_feedback_only':True})

def main():
    torch.set_num_threads(2);task=TASKS[0];out=ROOT/task/'data';out.mkdir(parents=True,exist_ok=True)
    if (out/'BANK.pt').exists():raise FileExistsError('P34 bank already sealed')
    origin=Path('evaluations/fast_adapter_p29_object_transfer_v01/tasks')/task/'data/meta'
    parts=[];sources=[];began=time.time();vis=None;reviews=[]
    for role in ('expert','base'):
        for path in sorted(origin.glob(f'{role}_[0-9]*.pt')):
            raw=torch.load(path,weights_only=True);meta=path.with_name(path.stem+'_EVALUATOR.json')
            configs=json.loads(meta.read_text())['configs'];envs=[ObjectEnv(c) for c in configs]
            try:
                for e,s in zip(envs,raw['reset_seeds']):e.reset(s)
                if vis is None:
                    cal=calibration(envs[0]);dump(ROOT/'CAMERA.json',cal);vis=RoleVisual(task,cal)
                frames=[]
                for t in range(201):
                    for j,e in enumerate(envs):
                        for k,v in raw['states'].items():getattr(e.env.data,k)[:]=v[j,t].numpy()
                        mujoco.mj_forward(e.env.model,e.env.data)
                    imgs=np.stack([e.rgb() for e in envs]);p=raw['proprio'][:,t].numpy()
                    if not parts and t in (0,60,100,160):
                        im=Image.fromarray(imgs[0]).resize((448,448));d=ImageDraw.Draw(im)
                        box=gripper_box(p[0],cal)
                        if box is not None:d.rectangle(tuple(2*x for x in box),outline='cyan',width=3)
                        d.text((5,5),f'Gripper ROI: measured EEF, t={t}',fill='white');reviews.append(im)
                        im.save(out/f'CROP_PREFLIGHT_{t}.jpg')
                    frames.append({k:v.cpu() for k,v in vis.encode(imgs,p).items()})
                b={k:torch.stack([f[k] for f in frames],1) for k in frames[0]}
                offset=int(path.stem.split('_')[-1]);n=len(configs)
                b.update(actions=raw['actions'].float(),proprio=raw['proprio'].float(),
                         object_ids=torch.arange(offset,offset+n),role=torch.full((n,),role=='base',dtype=torch.long))
                dest=out/path.name;torch.save(b,dest);parts.append(b);sources+=bindings([path,meta,dest])
                print('P34_DATA',role,offset,round(time.time()-began,1),flush=True)
            finally:
                for e in envs:e.close()
    torch.save({k:torch.cat([x[k] for x in parts]) for k in parts[0]},out/'BANK.pt')
    image=Image.new('RGB',(448*4,448))
    for i,im in enumerate(reviews):image.paste(im,(448*i,0))
    image.save(out/'CROP_AUDIT.jpg')
    for task in TASKS:prepare_stats(task)
    dump(ROOT/'DATA_RECEIPT.json',{'seconds':time.time()-began,'episodes_per_task':128,'training_objects':48,
         'validation_objects':16,'no_new_demonstrations':True,'no_teacher_labels':True,'sources':sources,
         'outputs':bindings([out/'BANK.pt',ROOT/'CAMERA.json']+[stats_path(t) for t in TASKS])})
    print('P34_DATA_COMPLETE',flush=True)

if __name__=='__main__':main()
