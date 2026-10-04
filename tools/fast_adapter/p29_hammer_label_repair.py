"""Correct stale expert phase on already grasped on-policy training states only.
All original trajectories/labels are preserved. No new rollout or test data used.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from tools.fast_adapter.p29_common import task_root,save_json,digest,ObjectEnv,HORIZON
from tools.fast_adapter.p29_hammer_repair import relation_teacher


def run():
    task=task_root('hammer-block');out=task/'data/meta_relabel_r3';out.mkdir(parents=True,exist_ok=False);records=[]
    total=changed=opens=0
    for p in sorted((task/'data/meta').glob('base_[0-9]*.pt')):
        b=torch.load(p,weights_only=True);b['teacher']=b['teacher'].clone()
        side=json.loads(p.with_name(p.stem+'_EVALUATOR.json').read_text());n=fix=bad=0
        for j,cfg in enumerate(side['configs']):
            e=ObjectEnv(cfg,render=False);e.reset(b['reset_seeds'][j])
            for t in range(HORIZON):
                for k in ('qpos','qvel','mocap_pos','mocap_quat'):getattr(e.env.data,k)[:]=b['states'][k][j,t].numpy()
                mujoco.mj_forward(e.env.model,e.env.data)
                left,right,*_=e.env.contacts();n+=1
                if left and right:
                    fix+=1;bad+=int(b['teacher'][j,t,3]<0)
                    b['teacher'][j,t]=torch.tensor(relation_teacher(e))
            e.close()
        dst=out/p.name;torch.save(b,dst)
        records.append({'source':str(p),'source_sha256':digest(p),'path':str(dst),'sha256':digest(dst),'frames':n,'held_frames_relabelled':fix,'stale_open_labels_corrected':bad})
        total+=n;changed+=fix;opens+=bad;print('RELABEL',p.name,'held',fix,'stale_open',bad,flush=True)
    save_json(out/'COMPLETE.json',{'time':time.time(),'all_original_data_preserved':True,'new_physical_episodes':0,'input_images_actions_states_unchanged':True,'only_teacher_changed':True,'expert_bank_unchanged':True,'frames':total,'held_frames_relabelled':changed,'stale_open_labels_corrected':opens,'files':records})
if __name__=='__main__':run()
