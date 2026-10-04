"""P64 existing-image cup cache and finite task training; no physics imports.

Cache creation is frozen DINO inference only. Training requires a separate
root-approved result-free freeze; neither command launches robot episodes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F

from prove.fast_adapter.p61_encoder import first_eef_anchor_transforms
from prove.fast_adapter.p61_data import sha256
from prove.fast_adapter.p60_contour import outer_hull_boundary, ContourMeasurement
from prove.fast_adapter.p64_cup import fixed_orientation, adapt, controller_estimate, fit_grasp_head

FRAMES = (0, 8, 16, 24, 29)
ENCODER_KEYS = ('dino_layers','role_patch_masks','role_valid','K','T_anchor_camera','T_anchor_eef')


def matrix(p, r):
    out=np.broadcast_to(np.eye(4,dtype=np.float32),(*np.shape(p)[:-1],4,4)).copy()
    out[...,:3,:3]=r;out[...,:3,3]=p
    return torch.tensor(out)


def geometry8(value, anchor):
    zeros=torch.zeros_like(value[...,:3])
    return fixed_orientation(torch.cat((value[...,:3],zeros,value[...,3:]),-1),anchor)


def materialize(source, output, model_dir, device='cuda'):
    from prove.fast_adapter.visual_alignment import FrozenDinoROI
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    episodes=sorted((source/'data_raw/cup/train').iterdir())
    if len(episodes)!=96:raise ValueError('Expected the existing 96 training configurations')
    vision=FrozenDinoROI(model_dir,device=device,microbatch=2)
    records=[];means=[];sizes=[]
    for episode in episodes:
        receipt=json.loads((episode/'RECEIPT.json').read_text())
        label=source/'data_labels/cup/train'/f'{episode.name}.npz'
        for path,expected in ((episode/'public_observations.npz',receipt['public_sha256']),
                              (episode/'visible_images.npz',receipt['images_sha256']),
                              (label,receipt['labels']['sha256'])):
            if sha256(path)!=expected:raise ValueError(f'Source hash changed: {path}')
        with np.load(episode/'public_observations.npz') as z:
            obs={k:z[k][list(FRAMES)].copy() for k in ('K','camera_position','camera_rotation','position','rotation')}
        with np.load(episode/'visible_images.npz') as z:
            rgb=z['rgb'][list(FRAMES)].copy()
            masks=np.unpackbits(z['role_masks_packed'][list(FRAMES)],axis=-1)[...,:448].astype(bool)
        eef=matrix(obs['position'],obs['rotation']);camera=matrix(obs['camera_position'],obs['camera_rotation'])
        anchor=eef[0];ac,ae=first_eef_anchor_transforms(anchor,camera,eef)
        layers=[]
        for start in range(0,len(FRAMES),2):
            with torch.inference_mode():
                pixels=torch.from_numpy(rgb[start:start+2]).to(device).permute(0,3,1,2).float()/255
                hidden=vision.model(pixel_values=((pixels-vision.mean)/vision.std).to(torch.bfloat16),output_hidden_states=True,return_dict=True).hidden_states
                layers.append(torch.stack([hidden[i][:,vision.prefix:] for i in (-12,-8,-4)],1).cpu().to(torch.float16))
        boundaries=[];hulls=[];valid=[]
        for mask in masks[:,2]:
            boundary,info=outer_hull_boundary(mask)
            valid.append(boundary is not None)
            boundaries.append(np.zeros((32,2),np.float32) if boundary is None else boundary)
            hulls.append(torch.tensor(info.get('hull_vertices',[]),dtype=torch.float32).reshape(-1,2))
        public=dict(dino_layers=torch.cat(layers)[:,None],role_patch_masks=F.avg_pool2d(torch.tensor(masks,dtype=torch.float32),16,16)[:,None],
            role_valid=torch.tensor(masks.reshape(len(FRAMES),3,-1).any(-1))[:,None],K=torch.tensor(obs['K'])[:,None],
            T_anchor_camera=ac[:,None],T_anchor_eef=ae,T_world_anchor=anchor,
            boundary_uv=torch.tensor(np.stack(boundaries)),boundary_valid=torch.tensor(valid),
            goal_hull_vertices=hulls,image_hw=torch.tensor([448,448]),frame_index=torch.tensor(FRAMES))
        with np.load(label) as z:
            centers=torch.tensor(z['center'][list(FRAMES)],dtype=torch.float32)
            half=torch.tensor(z['size'][list(FRAMES)],dtype=torch.float32)
        local=(centers-anchor[:3,3])@anchor[:3,:3]
        target=torch.cat((local,half.log()),-1)
        dest=output/episode.name;dest.mkdir()
        torch.save(public,dest/'public.pt');torch.save(dict(geometry5=target,half_sizes=half),dest/'private.pt')
        means.append(target);sizes.append(half)
        records.append(dict(path=str(dest.resolve()),seed=int(episode.name),source_receipt_sha256=sha256(episode/'RECEIPT.json'),
            public_sha256=sha256(dest/'public.pt'),private_sha256=sha256(dest/'private.pt')))
        print(json.dumps(dict(cached=len(records),total=96)),flush=True)
    manifest=dict(schema='prove.p64-cup-dino-cache.v1',records=records,frames=list(FRAMES),
        roles=['actor_robot','cup_body_and_support_context','functional_handle_front'],
        geometry_mean=torch.cat(means).mean(0).tolist(),geometry='anchor center3 + log OUTER halfsize2',
        training_yaw_degrees=[-10,10],evaluation_yaw_degrees=0,physics_steps=0,
        model_dir=str(Path(model_dir).resolve()),dino_sha256=sha256(Path(model_dir)/'model.safetensors'))
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def train_action(source, output, freeze):
    from prove.fast_adapter.p64_cup_policy import train_action_policy
    source,output,path=Path(source),Path(output),Path(freeze)
    contract=json.loads(path.read_text());review=json.loads((path.parent/'ROOT_TRAINING_REVIEW.json').read_text())
    if (review.get('approved') is not True or review.get('reviewer')!='root' or review.get('freeze_sha256')!=sha256(path)
            or contract.get('schema')!='prove.p64-cup-action-freeze.v1'
            or contract.get('training')!=dict(steps=1000,seed=11,batch_size=256,learning_rate=.003)):
        raise ValueError('Exact root-approved action freeze required')
    for name,digest in contract['files'].items():
        if sha256(name)!=digest:raise ValueError('Changed frozen source: '+name)
    if output.exists():raise FileExistsError(output)
    centers=[];sizes=[];bindings=[]
    labels=sorted((source/'data_labels/cup/train').glob('*.npz'))
    if len(labels)!=96:raise ValueError('Train-only 96 configurations required')
    for label in labels:
        receipt=source/'data_raw/cup/train'/label.stem/'RECEIPT.json'
        recorded=json.loads(receipt.read_text())
        if recorded['labels']['sha256']!=sha256(label):raise ValueError('Cup label hash mismatch')
        if str(label.resolve()) not in {str(Path(p).resolve()) for p in contract['files']}:raise ValueError('Every train label must be frozen')
        with np.load(label) as z:centers.append(z['center']);sizes.append(z['size'])
        bindings.append(dict(label=str(label),sha256=sha256(label)))
    model,metrics=train_action_policy(np.concatenate(centers),np.concatenate(sizes))
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='prove.p64-cup-action.v1',state_dict=model.state_dict(),metrics=metrics,
        training_freeze_sha256=sha256(path),sources=bindings),output)
    return metrics


def measurements(public, indices):
    from tools.fast_adapter.p61_train import contour_measurements
    return contour_measurements(public,indices)


def train(cache, output, initial_checkpoint, action_checkpoint, freeze, device='cuda'):
    from prove.fast_adapter.p64_common import P64RoleEncoder, train_encoder
    from prove.fast_adapter.p64_cup_policy import CupActionPolicy
    cache,output=Path(cache),Path(output)
    manifest=json.loads((cache/'MANIFEST.json').read_text())
    payload=torch.load(initial_checkpoint,map_location='cpu',weights_only=True)
    constructor=payload['constructor']
    torch.manual_seed(11)
    model=P64RoleEncoder(constructor['lila_source'],constructor['lila_checkpoint'],task='cup',
        geometry_mean=manifest['geometry_mean'],p61_checkpoint=initial_checkpoint,finetune_lila=True)
    head=CupActionPolicy().to(device);head.load_state_dict(torch.load(action_checkpoint,weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    def loss_callback(model,case,phase):
        path=Path(case['path'])
        public=torch.load(path/'public.pt',weights_only=True,mmap=True)
        private=torch.load(path/'private.pt',weights_only=True)
        indices=random.sample(range(len(FRAMES)),2) if phase=='supervised' else [2,3,4]
        kwargs={k:public[k][indices].to(device) for k in ENCODER_KEYS}
        kwargs={k:v if v.dtype==torch.bool else v.float() for k,v in kwargs.items()}
        prediction=model(**kwargs,image_hw=(448,448)).geometry
        anchor=public['T_world_anchor'].to(device)
        target=private['geometry5'][indices].to(device)
        def loss(pred,truth,z):
            corrected=torch.cat((pred[:3]+.02*z[:3],pred[3:]+.25*z[6:]),-1)
            geometry_error=((corrected-truth)/pred.new_tensor([.02,.02,.02,.25,.25])).square().mean()
            world_center=anchor[:3,:3]@corrected[:3]+anchor[:3,3]
            true_center=anchor[:3,:3]@truth[:3]+anchor[:3,3]
            # Use the identical frozen runtime action MLP and the same public
            # own/start pose in both predictions. Teacher geometry is isolated.
            start=anchor[:3,3][None].expand(3,-1)
            own=public['T_anchor_eef'][indices[-1],:3,3].to(device)
            own=(anchor[:3,:3]@own+anchor[:3,3])[None].expand(3,-1)
            phases=torch.tensor([1,2,3],device=device)  # align, descend, approach
            p=head(world_center[None].expand(3,-1),corrected[3:].exp()[None].expand(3,-1),start,own,phases)
            q=head(true_center[None].expand(3,-1),truth[3:].exp()[None].expand(3,-1),start,own,phases)
            return geometry_error+((p-q)/.02).square().mean()
        z=torch.zeros(8,device=device)
        if phase=='supervised':return sum(loss(p,t,z) for p,t in zip(prediction,target))/2
        for i,history in enumerate(([0,1,2],[0,1,2,3])):
            z,_=adapt(geometry8(prediction[i],anchor),z,anchor,measurements(public,history),arm='translation_size5',create_graph=True)
        return loss(prediction[-1],target[-1],z)+.25*loss(prediction[-1],target[-1],torch.zeros_like(z))
    return train_encoder(model,manifest['records'],loss_callback,checkpoint_path=output,freeze_path=freeze,
        bindings=dict(cache={'manifest_sha256':sha256(cache/'MANIFEST.json')},
                      source={'driver_sha256':sha256(__file__)},
                      checkpoint={'initial_sha256':sha256(initial_checkpoint),'action_head_sha256':sha256(action_checkpoint)}),device=device)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('cache');c.add_argument('--source',required=True);c.add_argument('--output',required=True);c.add_argument('--model',required=True);c.add_argument('--device',default='cuda')
    t=s.add_parser('train');t.add_argument('--cache',required=True);t.add_argument('--output',required=True);t.add_argument('--initial-checkpoint',required=True);t.add_argument('--action-checkpoint',required=True);t.add_argument('--freeze',required=True);t.add_argument('--device',default='cuda')
    a=s.add_parser('action');a.add_argument('--source',required=True);a.add_argument('--output',required=True);a.add_argument('--freeze',required=True)
    a=p.parse_args()
    if a.command=='cache':materialize(a.source,a.output,a.model,a.device)
    elif a.command=='train':train(a.cache,a.output,a.initial_checkpoint,a.action_checkpoint,a.freeze,a.device)
    else:train_action(a.source,a.output,a.freeze)
