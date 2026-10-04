#!/usr/bin/env python3
"""Offline BC warmup then differentiable meta-BC on native stage trajectories.

Input bundle: episodes list, each with public features [T+1,...], executed_action
[T,6], expert_action [T,6], episode_id and split. Features come from the frozen
frontend; expert labels never enter model.encode/feedback_loss. Parent must
freeze data, split, config and budgets before invoking a nonzero training run.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy


def frame(features,index):
    return {k:v[index] for k,v in features.items()}


def validate(episodes):
    ids=set()
    for e in episodes:
        if e['episode_id'] in ids:raise ValueError('Duplicate episode id')
        ids.add(e['episode_id'])
        a=e['executed_action'];label=e['expert_action'];f=e['features']
        if a.ndim!=2 or a.shape[-1]!=6 or label.shape!=a.shape:
            raise ValueError('Require distinct executed/expert OSC6 sequences')
        if len(a)<2 or any(len(v)!=len(a)+1 for v in f.values()):
            raise ValueError('Need >=2 consecutive transitions and T+1 public frames')
        if set(f)!={'raw','boxes','visible','proprio'}:
            raise ValueError('Only declared public features accepted')
        if not all(torch.isfinite(v).all() for v in (*f.values(),a,label)):
            raise ValueError('Nonfinite trajectory')
        if e['split'] not in ('train','validation','test'):raise ValueError('Explicit split required')


def batch(episodes,batch_size,generator,length=16):
    selected=[]
    for _ in range(batch_size):
        e=episodes[int(torch.randint(len(episodes),(),generator=generator))]
        if len(e['executed_action'])<length:raise ValueError('Training requires native segments >=16 transitions')
        t=int(torch.randint(len(e['executed_action'])-length+1,(),generator=generator))
        selected.append((e,t))
    features={k:torch.stack([e['features'][k][t:t+length+1] for e,t in selected]) for k in episodes[0]['features']}
    actions=torch.stack([e['executed_action'][t:t+length] for e,t in selected])
    labels=torch.stack([e['expert_action'][t:t+length] for e,t in selected])
    return features,actions,labels


def fit_normalization(model,episodes):
    """Training-trajectory statistics only; visible samples only for role groups."""
    features={k:torch.cat([e['features'][k] for e in episodes]) for k in episodes[0]['features']}
    visible=features['visible'].bool()
    with torch.no_grad():
        for key,mean,scale in [('raw',model.raw_mean,model.raw_scale),('boxes',model.box_mean,model.box_scale)]:
            for v in range(model.config.views):
                for r in range(3):
                    x=features[key][:,v,r][visible[:,v,r]]
                    if len(x)<2:raise ValueError(f'Insufficient visible training frames for view{v} role{r}')
                    mean[v,r].copy_(x.mean(0));scale[v,r].copy_(x.std(0).clamp_min(.01))
        x=features['proprio'];model.proprio_mean.copy_(x.mean(0));model.proprio_scale.copy_(x.std(0).clamp_min(.01))


def segment_loss(model,features,actions,labels,*,meta,prediction_weight=.1,ordinary=False):
    at=lambda t:{k:v[:,t] for k,v in features.items()}
    z0=model.initial(at(0));z=z0;history=[];bc=[];prediction=[]
    for t in range(actions.shape[1]):
        obs,nxt=at(t),at(t+1)
        # First eight teacher transitions support adaptation; later actions query.
        if not meta or t>=8:
            bc.append((model.action(obs,z)-labels[:,t]).square().mean())
        prediction.append(model.feedback_loss(obs,actions[:,t],nxt,z))
        history.append((obs,actions[:,t],nxt));history=history[-model.config.history:]
        if meta and t<actions.shape[1]-1:
            z,_=model.adapt_history(history,z,z0,create_graph=True)
    bc=torch.stack(bc).mean();pred=torch.stack(prediction).mean()
    return bc+(0 if ordinary else prediction_weight)*pred,dict(bc=bc,prediction=pred)


def train(episodes,config,*,seed,warmup_steps,joint_steps,batch_size,lr,prediction_weight,ordinary=False,device="cpu"):
    validate(episodes)
    train_episodes=[e for e in episodes if e['split']=='train']
    if not train_episodes:raise ValueError('No training episodes')
    torch.manual_seed(seed);g=torch.Generator().manual_seed(seed)
    model=StagePolicy(config);fit_normalization(model,train_episodes);model.to(device)
    train_episodes=[dict(e,features={k:v.to(device) for k,v in e["features"].items()},
                        executed_action=e["executed_action"].to(device),expert_action=e["expert_action"].to(device)) for e in train_episodes]
    start=time.monotonic()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr);last={}
    for step in range(warmup_steps+joint_steps):
        features,actions,labels=batch(train_episodes,batch_size,g)
        optimizer.zero_grad(set_to_none=True)
        loss,parts=segment_loss(model,features,actions,labels,
            meta=step>=warmup_steps and not ordinary,prediction_weight=prediction_weight,ordinary=ordinary)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();optimizer.step()
        last={k:float(v.detach()) for k,v in dict(loss=loss,**parts).items()}
        if (step+1)%100==0:print(json.dumps(dict(update=step+1,seconds=time.monotonic()-start,**last)),flush=True)
    last["seconds"]=time.monotonic()-start
    return model,last


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda','cuda:0','cuda:1'],default='cpu')
    args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError(args.output)
    settings=json.loads(args.config.read_text());config=Config(**settings['model'])
    data=torch.load(args.data,map_location='cpu',weights_only=True)
    model,last=train(data['episodes'],config,device=args.device,**settings['training'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='p52_stage_skill_v01',model=model.state_dict(),config=asdict(config),
                    settings=settings,data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    final_training_loss=last,claim='offline training only; no task-success claim'),args.output)
    print(json.dumps(last))


if __name__=='__main__':main()
