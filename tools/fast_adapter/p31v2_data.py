"""P31 TRAINING-only privileged supervision. Never imported by runtime control."""
import argparse
import json
import numpy as np
import torch
from tools.fast_adapter.p31v2_common import *
from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco
from prove.fast_adapter.continuous_adapter import inputs


def relation_labels(env):
    """Recoverable-state targets, no phase advancement from an unexecuted teacher.

    The .020 gripper-site offset, .18 lift goal, and15 action gain are inherited
    embodiment/controller calibration, not searched per-object motion scripts.
    """
    e = env.env
    p = e.tcp_center.copy()
    grasp = e.data.site('grip_site').xpos.copy(); grasp[2] -= .020
    left, right, *_ = e.contacts()
    bar, height = env.cfg['bar'], env.cfg['height']
    error = grasp-p
    near = bool(left and right) or np.linalg.norm(error / [2.5*bar, 2.5*bar, .85*height]) <= 1
    xy = np.linalg.norm(error[:2])
    if left and right:
        goal = p.copy(); goal[2] = e.start_z+.18; grip = .8
    elif xy <= .85*bar and abs(error[2]) <= .85*bar:
        goal = grasp; grip = .8
    else:
        goal = grasp.copy(); grip = -1.
        if xy > 2*bar:
            goal[2] = max(grasp[2]+1.5*height, .19)
    return np.r_[np.clip((goal-p)*15, -1, 1), grip].astype(np.float32), near


def raw_sources(role):
    return sorted((P29/TASK/'data/meta').glob(f'{role}_[0-9]*.pt'))


def load_original(path):
    b = torch.load(path, weights_only=True)
    configs = json.loads(path.with_name(path.stem+'_EVALUATOR.json').read_text())['configs']
    return b, configs


def label_record(b, configs):
    teacher = torch.zeros_like(b['actions']); near = torch.zeros(b['actions'].shape[:2], dtype=torch.bool)
    for j, c in enumerate(configs):
        env = ObjectEnv(c, render=False)
        try:
            env.reset(b['reset_seeds'][j])
            for t in range(b['actions'].shape[1]):
                for key, values in b['states'].items():
                    getattr(env.env.data, key)[:] = values[j, t].numpy()
                mujoco.mj_forward(env.env.model, env.env.data)
                a, active = relation_labels(env)
                teacher[j,t] = torch.from_numpy(a); near[j,t] = bool(active)
        finally:
            env.close()
    return {**{k:b[k] for k in ('visual','proprio','actions')}, 'teacher':teacher, 'near':near}


def qualify_teacher():
    b, configs = load_original(raw_sources('base')[0]); rows=[]
    for j, cfg in enumerate(configs):
        env=ObjectEnv(cfg, render=False)
        try:
            env.reset(b['reset_seeds'][j])
            aa=[]
            for t in range(200):
                a,_=relation_labels(env);aa.append(a);env.step(a)
            rows.append({'object':j,'success':any(x['success'] for x in env.metrics),
                         'contact_steps':sum(x['bilateral_handle'] for x in env.metrics),
                         'max_lift':max(x['lift'] for x in env.metrics)})
        finally:env.close()
    passed=sum(x['success'] for x in rows)>=protocol()['teacher_qualification']['minimum_successes']
    dump(ROOT/'TEACHER_QUALIFICATION.json',{'training_only':True,'rows':rows,'passed':passed,
          'bindings':bindings([Path(__file__),ROOT/'PROTOCOL.json'])})
    print('TEACHER_QUALIFICATION',passed,rows,flush=True)
    if not passed:raise RuntimeError('Training teacher did not qualify; no evaluation permitted')


def prepare():
    torch.set_num_threads(2)
    out=ROOT/'data';out.mkdir(parents=True,exist_ok=True)
    if (out/'old.pt').exists():raise FileExistsError('Training bank already exists')
    if not (ROOT/'TEACHER_QUALIFICATION.json').exists():qualify_teacher()
    assert json.loads((ROOT/'TEACHER_QUALIFICATION.json').read_text())['passed']
    parts=[];sources=[]
    for role in ('expert','base'):
        for path in raw_sources(role):
            b,configs=load_original(path)
            parts.append(label_record(b,configs));sources += [path,path.with_name(path.stem+'_EVALUATOR.json')]
            print('LABEL_OLD',path.name,flush=True)
    bank={k:torch.cat([x[k] for x in parts]) for k in FIELDS}
    assert len(bank['actions'])==128
    torch.save(bank,out/'old.pt')
    dump(out/'OLD_RECEIPT.json',{'episodes':128,'original_objects':64,'bindings':bindings(sources),
         'output_sha256':digest(out/'old.pt'),'teacher_consistent_between_arms':True})
    train_observer(bank)


def train_observer(bank):
    torch.manual_seed(protocol()['observer']['seed'])
    b={k:v.cuda() for k,v in bank.items()}
    prev=torch.cat((torch.zeros_like(b['actions'][:,:1]),b['actions'][:,:-1]),1)
    fit=torch.tensor(list(range(48))+list(range(64,112)),device='cuda')
    val=torch.tensor(list(range(48,64))+list(range(112,128)),device='cuda')
    net=StageObserver().cuda()
    x=inputs(b['visual'][fit,:-1],b['proprio'][fit,:-1],prev[fit])
    net.mean.copy_(x.mean((0,1)));net.scale.copy_(x.std((0,1)).clamp_min(.05))
    opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01)
    y=b['near'][fit].float();weight=((1-y).sum()/y.sum()).clamp(.25,4)
    for step in range(protocol()['observer']['steps']):
        ii=fit[torch.randint(len(fit),(512,),device='cuda')];tt=torch.randint(200,(512,),device='cuda')
        logits=net(b['visual'][ii,tt],b['proprio'][ii,tt],prev[ii,tt])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,b['near'][ii,tt].float(),pos_weight=weight)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    net.eval().requires_grad_(False)
    with torch.no_grad():
        pred=net.active(b['visual'][val,:-1],b['proprio'][val,:-1],prev[val]);truth=b['near'][val]
        recall=float((pred&truth).sum()/truth.sum());spec=float((~pred&~truth).sum()/(~truth).sum())
        q=protocol()['observer'];passed=recall>=q['minimum_recall'] and spec>=q['minimum_specificity']
    torch.save({'state':net.cpu().state_dict(),'fit_objects':list(range(48))},ROOT/'observer.pt')
    dump(ROOT/'OBSERVER_QUALIFICATION.json',{'passed':passed,'recall':recall,'specificity':spec,
         'qualification_training_objects':list(range(48,64)),'episodes':32,'frames':int(truth.numel()),
         'threshold':.5,'inputs':'visual+proprio+previous action only','bank_sha256':digest(ROOT/'data/old.pt'),
         'model_sha256':digest(ROOT/'observer.pt'),'steps':q['steps']})
    print('OBSERVER_QUALIFICATION',passed,recall,spec,flush=True)
    if not passed:raise RuntimeError('Observer qualification failed; stop stage experiment')


def collect_own(dino):
    # Frozen P30 parents control; evaluator/teacher sees states only after save.
    from tools.fast_adapter.p30v2_data import collect_block
    from tools.fast_adapter.p29_common import Visual
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    out=ROOT/'data/own_raw';out.mkdir(parents=True,exist_ok=False)
    visual=Visual(dino,TASK,torch.load(P29/TASK/'data/PCA.pt',weights_only=True))
    base=base_model(TASK);parts=[];sources=[]
    for index,path in enumerate(raw_sources('base')):
        old,configs=load_original(path);seed=11 if index<4 else 22
        parent=OLD/TASK/'models'/f'seed{seed}'/'gradient.pt'
        net=ObjectAdapter().cuda();net.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True)['state'])
        net.eval().requires_grad_(False);before=state_hash(net)
        b,metrics,_=collect_block(configs,old['reset_seeds'],visual,'evaluate',base,net,'gradient')
        assert before==state_hash(net)
        dest=out/f'own_{index*8:03d}.pt';torch.save(b,dest)
        dump(dest.with_suffix('.json'),{'configs':configs,'parent_seed':seed,'parent_sha256':digest(parent),
             'metrics':metrics,'trajectory_sha256':digest(dest),'all_failures_retained':True})
        parts.append(label_record(b,configs));sources += [dest,dest.with_suffix('.json'),parent]
        print('OWN_COLLECT',index+1,'/8',sum(any(x['success'] for x in m) for m in metrics),flush=True)
    original=torch.load(ROOT/'data/old.pt',weights_only=True)
    bank={k:torch.cat([original[k][:64]]+[x[k] for x in parts]) for k in FIELDS}
    torch.save(bank,ROOT/'data/own.pt')
    dump(ROOT/'data/OWN_RECEIPT.json',{'episodes':128,'new_episodes':64,'new_control_steps':12800,
         'bindings':bindings(sources+[ROOT/'data/old.pt']), 'output_sha256':digest(ROOT/'data/own.pt')})
    print('OWN_BANK_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect']);p.add_argument('--dino',default='assets/dinov3');a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:collect_own(a.dino)
