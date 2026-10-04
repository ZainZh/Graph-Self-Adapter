#!/usr/bin/env python3
"""Post-hoc read-only P52 control/adaptation diagnosis. No optimizer or simulator.

Opened train/validation and local rollout records are development evidence. Same-
state BC comparisons are counterfactual model outputs, not oracle action labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy
from tools.task_graph.p52_train import frame,segment_loss


def load_model(path):
    package=torch.load(path,map_location='cpu',weights_only=True)
    model=StagePolicy(Config(**package['config']));model.load_state_dict(package['model']);model.eval().requires_grad_(False)
    return model,package


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def summary(values):
    a=np.asarray(values,dtype=float)
    return dict(count=int(a.size),mean=float(a.mean()),median=float(np.median(a)),max=float(a.max()),first=float(a.flat[0]),last=float(a.flat[-1])) if a.size else None


def stack(rows):return {k:torch.stack([r[k] for r in rows]) for k in rows[0]}


def logs(folder):
    result={}
    for name in ('bc','joint'):
        rows=[]
        for line in (folder/'logs'/f'{name}.log').read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'update' in r:rows.append(r)
        result[name]=dict(rows=rows,final_ten={key:summary([r[key] for r in rows[-10:]]) for key in ('bc','prediction')},
                          final_ten_aux_to_bc=summary([.1*r['prediction']/max(r['bc'],1e-30) for r in rows[-10:]]))
    return result


def teacher_errors(models,episodes):
    rows=[]
    for e in episodes:
        f=e['features'];labels=e['expert_action'];count=len(labels)
        with torch.no_grad():
            fixed={name:model.action({k:v[:count] for k,v in f.items()},model.z0.expand(count,-1)) for name,model in models.items()}
        model=models['joint'];z0=model.z0.detach();z=z0.clone();history=[];online=[];znorm=[]
        for t in range(count):
            obs=frame(f,t)
            with torch.no_grad():online.append(model.action(obs,z));znorm.append(float((z-z0).norm()))
            history.append((obs,e['executed_action'][t],frame(f,t+1)));history=history[-8:]
            z,_=model.adapt_history(history,z,z0);z=z.detach()
        predictions=dict(bc=fixed['bc'],joint_fixed=fixed['joint'],joint_online=torch.stack(online))
        errors={name:(pred-labels).square().mean(-1).numpy() for name,pred in predictions.items()}
        rows.append(dict(episode_id=e['episode_id'],split=e['split'],transitions=count,
                         errors={name:dict(all=summary(err),first8=summary(err[:8]),later=summary(err[8:]),per_frame=err.tolist()) for name,err in errors.items()},
                         online_z_norm=summary(znorm),same_observation_online_fixed_action_norm=summary((predictions['joint_online']-predictions['joint_fixed']).norm(dim=-1).numpy())))
        print(json.dumps(dict(teacher_episode=e['episode_id'],split=e['split'],mse={name:float(err.mean()) for name,err in errors.items()})),flush=True)
    aggregate={}
    for split in ('train','validation'):
        selected=[r for r in rows if r['split']==split]
        aggregate[split]={name:{part:summary([v for r in selected for v in (r['errors'][name]['per_frame'][:8] if part=='first8' else r['errors'][name]['per_frame'][8:] if part=='later' else r['errors'][name]['per_frame'])]) for part in ('all','first8','later')} for name in ('bc','joint_fixed','joint_online')}
    return dict(episodes=rows,aggregate=aggregate)


def gradient_probe(model,episodes):
    selected=[e for e in episodes if e['split']=='train'][:4]
    features={k:torch.stack([e['features'][k][:17] for e in selected]) for k in selected[0]['features']}
    actions=torch.stack([e['executed_action'][:16] for e in selected]);labels=torch.stack([e['expert_action'][:16] for e in selected])
    model.requires_grad_(True)
    _,parts=segment_loss(model,features,actions,labels,meta=True,prediction_weight=.1)
    shared=list(model.role.parameters())+list(model.encoder.parameters())
    gb=torch.autograd.grad(parts['bc'],shared,retain_graph=True);gp=torch.autograd.grad(.1*parts['prediction'],shared)
    b=torch.cat([g.flatten() for g in gb]);p=torch.cat([g.flatten() for g in gp]);bn=b.norm();pn=p.norm()
    result=dict(scope='Post-hoc final-checkpoint gradient on first16 transitions of first4 declared train episodes; no optimizer update',episodes=[e['episode_id'] for e in selected],
                bc=float(parts['bc'].detach()),prediction=float(parts['prediction'].detach()),weighted_prediction=float(.1*parts['prediction'].detach()),
                shared_bc_gradient_norm=float(bn),shared_aux_gradient_norm=float(pn),aux_to_bc_gradient_ratio=float(pn/bn.clamp_min(1e-30)),
                gradient_cosine=float(torch.dot(b,p)/(bn*pn).clamp_min(1e-30)))
    model.requires_grad_(False);return result


def rollout_errors(models,root):
    cells=[]
    for result_path in sorted(root.rglob('RESULT.json')):
        result=json.loads(result_path.read_text())
        if 'cell' not in result or not (result_path.parent/'public/ACTIONS.jsonl').exists():continue
        folder=result_path.parent;records=[json.loads(s) for s in (folder/'public/ACTIONS.jsonl').read_text().splitlines()]
        feature_rows=[torch.load(folder/'public'/f'features_{r["t"]:04d}.pt',map_location='cpu',weights_only=True) for r in records]
        f=stack(feature_rows);count=len(records);joint=models['joint'];bc=models['bc']
        z=(joint.z0.detach().expand(count,-1).clone() if result['cell']['arm']=='bc' else torch.tensor([r['z_after'] for r in records],dtype=torch.float32))
        with torch.no_grad():
            a_bc=bc.action(f,bc.z0.expand(count,-1));a_fixed=joint.action(f,joint.z0.expand(count,-1));a_online=joint.action(f,z)
        rows=[];history=[]
        for i,(r,obs) in enumerate(zip(records,feature_rows)):
            loss_before=loss_after=None
            if i:
                history.append((feature_rows[i-1],torch.tensor(records[i-1]['action'][:6]),obs));history=history[-8:]
                ho=stack([h[0] for h in history]);ha=torch.stack([h[1] for h in history]);hn=stack([h[2] for h in history])
                before=torch.tensor(r['z_before']);after=torch.tensor(r['z_after'])
                with torch.no_grad():
                    loss_before=float(joint.feedback_loss(ho,ha,hn,before.expand(len(history),-1)) + joint.config.anchor*(before-joint.z0).square().mean())
                    loss_after=float(joint.feedback_loss(ho,ha,hn,after.expand(len(history),-1)) + joint.config.anchor*(after-joint.z0).square().mean())
            rows.append(dict(t=r['t'],executed=r['executed'],z_from_joint_initial=float((z[i]-joint.z0).norm()),
                             current_window_loss_before=loss_before,current_window_loss_after=loss_after,
                             recorded_inner_loss=r['inner_loss'],bc_action=a_bc[i].tolist(),joint_fixed_action=a_fixed[i].tolist(),joint_recorded_z_action=a_online[i].tolist(),
                             recorded_action=r['action'][:6],proprio_zscore_rms=float(((obs['proprio']-joint.proprio_mean)/joint.proprio_scale).square().mean().sqrt())))
        online_delta=(a_online-a_fixed);between=(a_bc-a_fixed)
        losses=[r for r in rows[1:] if r['recorded_inner_loss'] is not None]
        action=np.asarray([r['action'][:6] for r in records if r['executed']])
        cells.append(dict(directory=str(folder),cell=result['cell'],result=result,
            same_state_bc_vs_joint_fixed=summary(between.norm(dim=-1).numpy()),
            same_state_joint_z_effect=summary(online_delta.norm(dim=-1).numpy()),
            same_state_joint_z_translation_m=summary((online_delta[:,:3].norm(dim=-1)*.05).numpy()),
            same_state_joint_z_rotation_rad=summary((online_delta[:,3:].norm(dim=-1)*.5).numpy()),
            z_drift=summary([r['z_from_joint_initial'] for r in rows]),
            inner=dict(updates=len(losses),decreased=sum(r['current_window_loss_after']<r['current_window_loss_before'] for r in losses),
                       before=summary([r['current_window_loss_before'] for r in losses]),after=summary([r['current_window_loss_after'] for r in losses])),
            normalized_action_max_per_axis=np.abs(action).max(0).tolist(),action_saturation_fraction=float((np.abs(action)>.95).mean()),
            proprio_zscore_rms=summary([r['proprio_zscore_rms'] for r in rows]),rows=rows))
        print(json.dumps(dict(cell=result['cell'],success=result['local_success'],z_drift=cells[-1]['z_drift']['max'],z_action_effect=cells[-1]['same_state_joint_z_effect']['max'])),flush=True)
    return cells


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    post=a.root/'postprocess_v01';models={};packages={}
    for name in ('bc','joint'):models[name],packages[name]=load_model(post/f'{name}_final.pt')
    data=torch.load(post/'native_data.pt',map_location='cpu',weights_only=True)
    write(a.output/'TRAINING_LOGS.json',logs(post))
    write(a.output/'FINAL_CHECKPOINT_GRADIENT_PROBE.json',gradient_probe(models['joint'],data['episodes']))
    write(a.output/'TEACHER_ACTION_ERRORS.json',teacher_errors(models,data['episodes']))
    write(a.output/'ROLLOUT_MODEL_DIAGNOSTIC.json',rollout_errors(models,a.root/'automatic_eval_v01'))
    write(a.output/'SCOPE.json',dict(training=False,new_rollout=False,network_modified=False,
         evidence='Opened development records, counterfactual actions and final-checkpoint gradients only. Loss decline is not task success.',
         teacher_query='Full teacher trajectory persistent z, no reset every16; offline teacher-forced, not closed loop',
         training_design_facts=['Meta BC only t8..15 in sampled16; global episode t0..7 never query-supervised after warmup',
           'Shared encoder receives BC and 0.1 prediction gradients','Training segments restart at commonz0; deployment persists up to180 observed actions']))


if __name__=='__main__':main()
