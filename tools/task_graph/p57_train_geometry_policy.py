#!/usr/bin/env python3
"""P57 B offline full-stage BC/meta training with frozen A2 public geometry."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from prove.task_graph.p57_geometry_policy import (GeometryPolicy,PolicyConfig,action_context,
    instant_geometry,predict_public_uv,prior_frames,public_rays)
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p57_train_measurement import load_bundle,public_input,validate_bundle


ROOT=Path(__file__).resolve().parents[2]
SOURCE_ROOTS=('prove/task_graph/p57_geometry_policy.py',
              'tools/task_graph/p57_train_geometry_policy.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def source_hashes():
    return {name:sha(ROOT/name) for name in source_closure(ROOT,SOURCE_ROOTS)}


def reference(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('P57 B frozen input must be inside repository')
    return dict(path=str(path),repo_relative=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def verify_reference(ref, supplied=None):
    relative=Path(ref['repo_relative'])
    mapped=(ROOT/relative).resolve()
    if relative.is_absolute() or not mapped.is_relative_to(ROOT):
        raise ValueError('Unsafe P57 B frozen repository reference')
    candidate=Path(supplied).resolve() if supplied is not None else mapped
    if sha(candidate)!=ref['sha256']:
        raise ValueError('P57 B frozen reference changed')
    return candidate


def freeze(manifest_path,a2_path,plan_path,output):
    manifest_path=Path(manifest_path).resolve();a2_path=Path(a2_path).resolve()
    plan_path=Path(plan_path).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    a2=torch.load(a2_path,map_location='cpu',weights_only=True)
    if a2.get('schema')!='p57.a2-translation-readout.v1' or a2.get('mode')!='mask_geometry':
        raise ValueError('Require qualified frozen A2 mask origin readout')
    value=dict(schema='p57.geometry-policy-training-freeze.v1',
               plan=reference(plan_path),
               pack_manifest=reference(manifest_path),
               a2_head=reference(a2_path),
               sources=source_hashes(),config=asdict(PolicyConfig()),
               training=dict(seed=11,batch=4,warmup_updates=500,joint_updates=1000,
                             optimizer='Adam',learning_rate=3e-4,
                             sampling='uniform whole actual train trajectory',selection='final only',
                             warmup='fixed BC, B and z0 frozen',
                             joint='.5 fixed BC + .5 online BC, 3-step higher-order inner'),
               output_relative='runs_b/HEAD.pt',physics_steps=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    return value


def verify(freeze_path,manifest_path,a2_path,output):
    freeze_path=Path(freeze_path).resolve();output=Path(output).resolve()
    value=json.loads(freeze_path.read_text())
    if value.get('schema')!='p57.geometry-policy-training-freeze.v1' or value.get('physics_steps')!=0:
        raise ValueError('Invalid P57 B training freeze')
    for key,path in (('pack_manifest',manifest_path),('a2_head',a2_path)):
        verify_reference(value[key],path)
    verify_reference(value['plan'])
    if value['sources']!=source_hashes():
        raise ValueError('P57 B plan or source changed')
    if value['config']!=asdict(PolicyConfig()) or value['training']!=dict(
            seed=11,batch=4,warmup_updates=500,joint_updates=1000,optimizer='Adam',learning_rate=3e-4,
            sampling='uniform whole actual train trajectory',selection='final only',
            warmup='fixed BC, B and z0 frozen',
            joint='.5 fixed BC + .5 online BC, 3-step higher-order inner'):
        raise ValueError('P57 B training configuration changed')
    if output!=(freeze_path.parent/value['output_relative']).resolve():
        raise ValueError('P57 B output outside frozen queue')
    return value


def prepare_public_episode(episode,readout):
    """Public-only geometry cache. Expert labels never enter this function."""
    _,_,R_held,R_goal=prior_frames(readout)
    n=len(episode['input']['features']['proprio'])-1
    data={key:[] for key in ('g','eef_pos','R_desired','dr','bboxwh','gripper','A','C','valid','uv')}
    for t in range(n):
        public=public_input(episode,t)
        uv,role_valid,public=predict_public_uv(readout,public)
        K=public['camera_K'].numpy();camera=public['camera_pose'].numpy();eef=public['eef_transform'].numpy()
        g,fit=instant_geometry(uv,role_valid,K,camera,eef)
        if g is None:
            raise ValueError(f'Public origin fit degenerate at {episode["condition_id"]} frame {t}')
        A,C,valid=public_rays(uv,role_valid,K,camera,eef)
        eef_pos,R_desired,dr,bboxwh,gripper=action_context(public,R_held,R_goal)
        for key,value in dict(g=torch.from_numpy(g),eef_pos=eef_pos,R_desired=R_desired,
                              dr=dr,bboxwh=bboxwh,gripper=gripper,A=A,C=C,valid=valid,
                              uv=torch.from_numpy(uv.astype('float32'))).items():
            data[key].append(value)
    return dict(condition_id=episode['condition_id'],source_policy=episode['source_policy'],
                source_seal_sha256=episode['source_seal_sha256'],
                public_frame_sha256=[row['sha256'] for row in episode['input']['public_frames'][:n]],
                **{key:torch.stack(value) for key,value in data.items()})


def stage_loss(model,public,expert,*,joint):
    T=len(expert)
    if T<1 or any(len(public[key])!=T for key in ('g','A','C','valid','eef_pos','dr','bboxwh','gripper')):
        raise ValueError('P57 B actual full-stage lengths differ')
    z0=model.z0;z=z0;history=[];fixed=[];online=[];latent=[]
    for t in range(T):
        history.append((public['A'][t],public['C'][t],public['valid'][t]))
        history=history[-model.config.history:]
        g=public['g'][t]
        if joint and t>=1:
            support=tuple(torch.stack([row[i] for row in history]) for i in range(3))
            z,_=model.adapt(g,z,support,create_graph=True)
        arguments=[public[key][t] for key in ('eef_pos','R_desired','dr','bboxwh','gripper')]
        fixed.append((model.action(g,z0,*arguments)-expert[t]).square().mean())
        online.append((model.action(g,z,*arguments)-expert[t]).square().mean())
        latent.append(z)
    fixed=torch.stack(fixed).mean();online=torch.stack(online).mean()
    return ((fixed+online)*.5 if joint else fixed),dict(fixed=fixed,online=online,
        initial_z=z0,final_z=z,latents=torch.stack(latent))


def train(manifest_path,a2_path,freeze_path,output,*,device='cpu'):
    freeze_path=Path(freeze_path);output=Path(output)
    verify(freeze_path,manifest_path,a2_path,output)
    report_path=output.parent/'REPORT.json';cache_path=output.parent/'PUBLIC_GEOMETRY_CACHE.pt'
    for path in (output,report_path,cache_path):
        if path.exists():raise FileExistsError(path)
    a2=torch.load(a2_path,map_location='cpu',weights_only=True)
    if a2.get('schema')!='p57.a2-translation-readout.v1' or a2.get('mode')!='mask_geometry':
        raise ValueError('Wrong frozen A2 readout')
    episodes=load_bundle(manifest_path)['episodes']
    train_indices,dev_indices=validate_bundle(dict(schema='p57.geometry-bundle.v1',episodes=episodes))
    public=[prepare_public_episode(episodes[index],a2) for index in train_indices]
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='p57.geometry-public-cache.v1',pack_manifest_sha256=sha(manifest_path),
                    a2_head_sha256=sha(a2_path),episodes=public,labels_included=False),cache_path)
    public=[{key:(item.to(device) if isinstance(item,torch.Tensor) else item) for key,item in episode.items()}
            for episode in public]
    expert=[episodes[index]['expert_actions'].to(device) for index in train_indices]
    torch.manual_seed(11);rng=torch.Generator().manual_seed(11)
    model=GeometryPolicy().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    trace=[]
    for phase,updates in (('warmup',500),('joint',1000)):
        model.B.requires_grad_(phase=='joint');model.z0.requires_grad_(phase=='joint')
        for step in range(updates):
            picks=torch.randint(len(public),(4,),generator=rng).tolist()
            optimizer.zero_grad(set_to_none=True)
            losses=[];parts=[]
            for index in picks:
                loss,info=stage_loss(model,public[index],expert[index],joint=phase=='joint')
                losses.append(loss);parts.append(info)
            mean=torch.stack(losses).mean()
            if not torch.isfinite(mean):raise RuntimeError('Nonfinite P57 B training loss')
            mean.backward();optimizer.step()
            if (step+1)%100==0:
                row=dict(phase=phase,update=step+1,loss=float(mean.detach()),
                         fixed=float(torch.stack([x['fixed'] for x in parts]).mean().detach()),
                         online=float(torch.stack([x['online'] for x in parts]).mean().detach()))
                trace.append(row);print(json.dumps(row),flush=True)
    model.B.requires_grad_(True);model.z0.requires_grad_(True)
    checkpoint=dict(schema='p57.geometry-policy.v1',config=asdict(model.config),state_dict=model.cpu().state_dict(),
                    a2_head_sha256=sha(a2_path),pack_manifest_sha256=sha(manifest_path),
                    freeze_sha256=sha(freeze_path),public_cache_sha256=sha(cache_path),
                    source_sha256=source_hashes(),training_trace=trace,
                    train_episodes=len(train_indices),development_episodes_unseen=len(dev_indices),
                    claim='offline local skill; no physical success or online gain yet')
    torch.save(checkpoint,output)
    report=dict(schema='p57.geometry-policy-training-report.v1',checkpoint_sha256=sha(output),
                public_cache_sha256=sha(cache_path),a2_head_sha256=sha(a2_path),
                pack_manifest_sha256=sha(manifest_path),freeze_sha256=sha(freeze_path),
                training_trace=trace,train_episodes=len(train_indices),
                development_episodes_unseen=len(dev_indices))
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(output=str(output),checkpoint_sha256=sha(output),report=str(report_path))),flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--manifest',type=Path,required=True);f.add_argument('--a2',type=Path,required=True);f.add_argument('--plan',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--manifest',type=Path,required=True);t.add_argument('--a2',type=Path,required=True);t.add_argument('--freeze',type=Path,required=True);t.add_argument('--output',type=Path,required=True);t.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(1)
    if args.command=='freeze':print(json.dumps(dict(output=str(args.output),schema=freeze(args.manifest,args.a2,args.plan,args.output)['schema'])))
    else:train(args.manifest,args.a2,args.freeze,args.output,device=args.device)


if __name__=='__main__':main()
