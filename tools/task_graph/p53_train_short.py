#!/usr/bin/env python3
"""P53 prepared training entrypoints; no physics or automatic model selection.

One common 1000-update warmup artifact is forked into BC and joint, each2000
updates. Whole native short stages retain variable lengths, with no padding or
midstage z resets. Horizon must be fixed externally before execution.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from prove.task_graph.p52_stage_policy import Config
from prove.task_graph.p53_short_insert import ShortInsertPolicy,full_stage_loss
from tools.task_graph.p52_train import fit_normalization


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def validate_dataset(data,horizon):
    if horizon!=24:raise ValueError('Root-approved P53 horizon is exactly24')
    if data.get('stage_scope')!='near_contact_insert':raise ValueError('Require explicit native near-contact Insert data scope')
    ids=set();counts=dict(train=0,validation=0,test=0)
    for episode in data['episodes']:
        if episode['episode_id'] in ids:raise ValueError('Duplicate episode identity')
        ids.add(episode['episode_id']);split=episode['split']
        if split not in counts:raise ValueError('Explicit trajectory split required')
        counts[split]+=1;length=len(episode['executed_action'])
        if not 1<=length<=horizon:raise ValueError('Native stage outside declared horizon; do not silently truncate')
        if episode['executed_action'].shape!=(length,6) or episode['expert_action'].shape!=(length,6):raise ValueError('OSC6 sequence mismatch')
        features=episode['features']
        if set(features)!={'raw','boxes','visible','proprio'} or any(len(v)!=length+1 for v in features.values()):raise ValueError('T+1 public features required')
        if any(not torch.isfinite(v).all() for v in [*features.values(),episode['executed_action'],episode['expert_action']]):raise ValueError('Nonfinite trajectory')
    if not counts['train']:raise ValueError('No native short-stage training trajectories')
    return counts


def sample_episodes(episodes,generator,count=4):
    # Uniform trajectory sampling, not random windows. Each sampled stage starts
    # at its actual first observation and includes its entire recorded duration.
    return [episodes[int(torch.randint(len(episodes),(),generator=generator))] for _ in range(count)]


def minibatch_loss(model,episodes,*,horizon,branch):
    if branch not in ('warmup','bc','joint'):raise ValueError('Unknown training branch')
    losses=[];metrics=[]
    for episode in episodes:
        loss,parts=full_stage_loss(model,episode['features'],episode['executed_action'],episode['expert_action'],horizon=horizon,
              meta=branch=='joint',fixed_bc_weight=.5,auxiliary_weight=0 if branch=='bc' else .1)
        losses.append(loss);metrics.append(parts)
    return torch.stack(losses).mean(),{key:torch.stack([m[key] for m in metrics]).mean() for key in ('control','online_bc','fixed_bc','auxiliary')}


def to_device(episodes,device):
    return [dict(e,features={k:v.to(device) for k,v in e['features'].items()},executed_action=e['executed_action'].to(device),
                 expert_action=e['expert_action'].to(device)) for e in episodes if e['split']=='train']


def optimize(model,optimizer,episodes,generator,*,horizon,branch,steps):
    start=time.monotonic();last={}
    for update in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss,parts=minibatch_loss(model,sample_episodes(episodes,generator),horizon=horizon,branch=branch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective; stop without retry')
        loss.backward();optimizer.step()
        last={k:float(v.detach()) for k,v in dict(loss=loss,**parts).items()}
        if (update+1)%100==0:print(json.dumps(dict(branch=branch,update=update+1,seconds=time.monotonic()-start,**last)),flush=True)
    return dict(seconds=time.monotonic()-start,last=last,updates=steps)


def create_warmup(data_path,output,horizon,device):
    if output.exists():raise FileExistsError(output)
    data=torch.load(data_path,map_location='cpu',weights_only=True);counts=validate_dataset(data,horizon)
    torch.manual_seed(11);generator=torch.Generator().manual_seed(11)
    model=ShortInsertPolicy(Config());fit_normalization(model,[e for e in data['episodes'] if e['split']=='train']);model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    result=optimize(model,optimizer,to_device(data['episodes'],device),generator,horizon=horizon,branch='warmup',steps=1000)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='p53_common_warmup_v01',model=model.state_dict(),optimizer=optimizer.state_dict(),config=asdict(model.config),
        generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),horizon=horizon,seed=11,data_sha256=sha(data_path),
        trajectory_splits=counts,common_outer_updates=1000,result=result,
        objective='all-time fixed-z BC + .1 predictor-only detached auxiliary'),output)


def fork_common(common_path,device='cpu'):
    common=torch.load(common_path,map_location='cpu',weights_only=True)
    if common['schema']!='p53_common_warmup_v01' or common['common_outer_updates']!=1000:raise ValueError('Wrong common warmup artifact')
    model=ShortInsertPolicy(Config(**common['config']));model.load_state_dict(common['model'],strict=True);model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);optimizer.load_state_dict(common['optimizer'])
    generator=torch.Generator();generator.set_state(common['generator_state']);torch.set_rng_state(common['torch_rng_state'])
    return model,optimizer,generator,common


def train_branch(data_path,common_path,output,branch,device):
    if output.exists():raise FileExistsError(output)
    model,optimizer,generator,common=fork_common(common_path,device)
    if sha(data_path)!=common['data_sha256']:raise ValueError('Branch data differ from shared warmup')
    data=torch.load(data_path,map_location='cpu',weights_only=True);validate_dataset(data,common['horizon'])
    if branch=='bc':model.predictor.requires_grad_(False)
    result=optimize(model,optimizer,to_device(data['episodes'],device),generator,horizon=common['horizon'],branch=branch,steps=2000)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='p53_short_insert_skill_v01',branch=branch,model=model.state_dict(),config=asdict(model.config),
        common_warmup_sha256=sha(common_path),data_sha256=common['data_sha256'],horizon=common['horizon'],seed=11,
        common_outer_updates=1000,branch_outer_updates=2000,total_outer_updates=3000,result=result,
        objective='all-time ordinary BC' if branch=='bc' else '.5 all-time online-z BC + .5 all-time fixed-z BC + .1 predictor-only auxiliary',
        checkpoint_selection='final only; no validation selection',deployment='all network weights frozen; only stage z updates'),output)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for command in ('warmup','branch'):
        q=sub.add_parser(command);q.add_argument('--data',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--device',default='cpu')
        if command=='warmup':q.add_argument('--horizon',type=int,required=True)
        else:q.add_argument('--common',type=Path,required=True);q.add_argument('--branch',choices=['bc','joint'],required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='warmup':create_warmup(a.data,a.output,a.horizon,a.device)
    else:train_branch(a.data,a.common,a.output,a.branch,a.device)


if __name__=='__main__':main()
