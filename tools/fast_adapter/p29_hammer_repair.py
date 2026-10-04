"""Development-only repair of hammer skill labels using physical relation feedback.
No changed contacts/success criterion, no added runtime simulator truth.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p29_common import *
from tools.fast_adapter.p29_data import collect_block
from prove.fast_adapter.object_interaction_adapter import ObjectSkill

TASK='hammer-block'

def relation_teacher(self):
    e=self.env;p=e.tcp_center.copy();g=e.data.site('grip_site').xpos.copy();tip=e.data.site('tool_site').xpos.copy();target=e.data.site('target_site').xpos.copy()
    grasp=g.copy();grasp[2]-=.005
    left,right,*_=e.contacts();lift=float(e.data.body('actor').xpos[2]-e.start_z)
    # Holding is a training label fact only. Geometry feedback replaces the old
    # irreversible phase-3 pre-strike waypoint and phase-counter transition.
    if (left and right) or lift>.025:
        grip=.9;desired=target.copy();desired[1]+=.055-float(e.data.joint('target_slide').qpos[0])
        if tip[2]<target[2]-.008:
            goal=p+np.array([0.,0.,target[2]+.006-tip[2]])
        else:goal=p+(desired-tip)
    else:
        xy=np.linalg.norm((p-grasp)[:2]);dz=abs(p[2]-grasp[2])
        if xy>.010:
            goal=grasp+np.array([0,0,.10]);grip=-1.
        elif dz>.010:
            goal=grasp;grip=-1.
        else:goal=grasp;grip=.9
    return np.r_[np.clip((goal-p)*15,-1,1),grip].astype(np.float32)

def qualify():
    out=task_root(TASK)/'candidates/base_v02/teacher_qualification';out.mkdir(parents=True,exist_ok=False);rows=[]
    for split in ('canonical','train','geometry_ood'):
        for i in range(8):
            e=ObjectEnv(object_config(TASK,2900100+i,split),render=False);e.reset(2900200+i)
            for _ in range(HORIZON):e.step(relation_teacher(e))
            rows.append({'split':split,'i':i,'success':bool(any(m['success'] for m in e.metrics)),'terminal_success':bool(e.metrics[-1]['success']),'head_contacts':sum(x['head_target_contact'] for x in e.metrics),'max_target':max(x['target_displacement'] for x in e.metrics)})
            e.close()
    save_json(out/'RESULT.json',{'rows':rows,'passed':all(sum(x['success'] for x in rows if x['split']==st)>=7 for st in ('canonical','train','geometry_ood'))})
    print(rows,flush=True)

def relabel():
    torch.set_num_threads(4);out=task_root(TASK)/'candidates/base_v02';out.mkdir(parents=True,exist_ok=True);banks=[];sources=[]
    for stage in ('demos','dagger'):
        for p in sorted((task_root(TASK)/'data'/stage).glob(f'{stage}_[0-9]*.pt')):
            b=torch.load(p,weights_only=True);side=json.loads(p.with_name(p.stem+'_EVALUATOR.json').read_text());labels=[]
            for j,cfg in enumerate(side['configs']):
                e=ObjectEnv(cfg,render=False);e.reset(b['reset_seeds'][j]);ts=[]
                for t in range(HORIZON):
                    for k in ('qpos','qvel','mocap_pos','mocap_quat'):getattr(e.env.data,k)[:]=b['states'][k][j,t].numpy()
                    import mujoco;mujoco.mj_forward(e.env.model,e.env.data);ts.append(relation_teacher(e))
                labels.append(ts);e.close()
            if stage=='demos':
                # Reuse the exact frozen training-only projection.
                pc=torch.load(task_root(TASK)/'data/PCA.pt',weights_only=True);proj=[]
                for f in b['features'].split(2):proj.append(((f.float().cuda()-pc['mean'].cuda())@pc['basis'].cuda()/pc['sd'].cuda()).cpu())
                v=torch.cat(proj)*b['presence'][...,None].float();v=torch.cat((v,b['boxes'].float(),b['presence'][...,None].float()),-1).flatten(-2)
            else:v=b['visual']
            banks.append({'visual':v,'proprio':b['proprio'],'actions':b['actions'],'teacher':torch.tensor(np.array(labels),dtype=torch.float32)})
            sources.append({'path':str(p),'sha256':digest(p)})
    b={k:torch.cat([x[k] for x in banks]) for k in banks[0]};torch.save(b,out/'RELABELLED.pt');save_json(out/'RELABEL_SEAL.json',{'sources':sources,'path':str(out/'RELABELLED.pt'),'sha256':digest(out/'RELABELLED.pt'),'method':'training-only relation teacher on previously collected states, no new held-out states','unique_episodes':len(b['actions'])});print('RELABELLED',len(b['actions']),flush=True)

def train():
    torch.set_num_threads(2);torch.manual_seed(2922);out=task_root(TASK)/'candidates/base_v02';b={k:v.cuda() for k,v in torch.load(out/'RELABELLED.pt',weights_only=True).items()}
    prev=torch.cat((torch.zeros_like(b['actions'][:,:1]),b['actions']),1);net=ObjectSkill().cuda();net.load_state_dict(torch.load(task_root(TASK)/'models/base_bc.pt',weights_only=True)['state']);opt=torch.optim.Adam(net.parameters(),lr=.0002)
    for step in range(8000):
        d=torch.randint(len(b['actions']),(512,),device='cuda');t=torch.randint(HORIZON,(512,),device='cuda');a=net(b['visual'][d,t],b['proprio'][d,t],prev[d,t]);loss=(a-b['teacher'][d,t]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%1000==0:print('repair',step,float(loss.detach()),flush=True)
    torch.save({'state':net.cpu().state_dict(),'updates':8000,'parent':'base_bc','label_source':'RELABELLED.pt'},out/'base.pt');save_json(out/'TRAIN_COMPLETE.json',{'path':str(out/'base.pt'),'sha256':digest(out/'base.pt'),'steps':8000})

def qualify_base(dino='assets/dinov3'):
    out=task_root(TASK)/'candidates/base_v02';net=ObjectSkill().cuda();net.load_state_dict(torch.load(out/'base.pt',weights_only=True)['state']);net.eval().requires_grad_(False)
    visual=Visual(dino,TASK,torch.load(task_root(TASK)/'data/PCA.pt',weights_only=True))
    cfgs=[object_config(TASK,2940000+i,'canonical') for i in range(8)];seeds=[2942000+i for i in range(8)]
    b,m=collect_block(TASK,cfgs,seeds,HORIZON,'base',visual,net,labels=False);path=out/'QUALIFICATION.pt';torch.save(b,path)
    successes=[bool(any(x['success'] for x in xs)) for xs in m]
    save_json(out/'BASE_GATE.json',{'passed':sum(successes)>=6,'successes':sum(successes),'n':8,'success':successes,'source_sha256':digest(path),'configs':cfgs,'metrics':m,'same_opened_development_resets':True})
    print('REPAIR_BASE_GATE',sum(successes),'/8',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','relabel','train','qualify_base']);a=p.parse_args();globals()[a.mode]()
