"""Bounded P43 meta-training; privileged labels are a separate trainer input."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.neural_geometry_adapter import (NeuralGeometry,observation_features,
    align_observation,ray_matrices,GEOMETRY_SCALE)
from prove.fast_adapter.geometry3d_adapter import initial,command,pose


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def load_data(observation_path,label_path,task):
    data=np.load(observation_path,allow_pickle=False);labels=np.load(label_path,allow_pickle=False)
    forbidden={'oracle','success','target','target_position','truth','target_corners'}&set(data.files)
    if forbidden:raise ValueError(f'Privileged runtime keys: {forbidden}')
    truth=labels['corners']
    episodes=[]
    for episode in range(data['corners'].shape[0]):
        rows=[];reference=None;first=None
        for frame in range(data['corners'].shape[1]):
            if not bool(data['corners_valid'][episode,frame]):continue
            obs={key:data[key][episode,frame] for key in ('corners','camera_position','camera_rotation','K','position','rotation','role_bbox','role_visible','local')}
            if reference is None:
                try:reference=initial(obs,task)
                except RuntimeError:continue
            obs=align_observation(obs,reference)
            target=truth[episode,frame] if truth.ndim==4 else truth[episode]
            # All candidates have canonical geometric correspondence; no GT is
            # used to align masks or inputs. Labels retain the canonical frame.
            rows.append(dict(feature=observation_features(obs),ray=ray_matrices(obs),truth=target,t=int(data['t'][episode,frame])))
        if len(rows)<4:continue
        episodes.append(dict(seed=int(data['seeds'][episode]),rows=rows,reference=reference.astype(np.float32)))
    return episodes


def batch_items(episodes,rng,batch_size,support_count,device):
    eligible=[e for e in episodes if len(e['rows'])>support_count]
    if not eligible:raise ValueError(f'No episode has {support_count} unique support views and a later query')
    selection=[eligible[int(rng.integers(len(eligible)))] for _ in range(batch_size)]
    reference=[];first=[];support=[];query=[];a=[];c=[];target=[]
    for episode in selection:
        rows=episode['rows'];query_index=int(rng.integers(support_count,len(rows)))
        indices=np.linspace(0,query_index-1,support_count).round().astype(int)
        picked=[rows[i] for i in indices]
        reference.append(episode['reference']);first.append(rows[0]['feature'])
        support.append([r['feature'] for r in picked]);query.append(rows[query_index]['feature'])
        a.append([r['ray'][0] for r in picked]);c.append([r['ray'][1] for r in picked]);target.append(rows[query_index]['truth'])
    tensor=lambda x:torch.tensor(np.array(x),dtype=torch.float32,device=device)
    return [tensor(x) for x in (reference,first,support,query,a,c,target)]


def meta_prediction(model,batch,create_graph):
    reference,first,support,query,A,C,target=batch
    z0,_=model.encode(first,first,reference);z=z0
    inner=[]
    for s in range(support.shape[1]):
        _,offset=model.encode(first,support[:,s],reference)
        if s>=2:
            z,records=model.adapt(reference,offset,z,A[:,:s+1],C[:,:s+1],create_graph=create_graph)
            inner.extend(records)
    _,offset=model.encode(first,query,reference)
    return model.decode(reference,offset,z0),model.decode(reference,offset,z),z0,z,inner


def evaluation(model,episodes,device):
    model.eval();rng=np.random.default_rng(91212);values=[];dependence=[]
    for _ in range(8):
        batch=batch_items(episodes,rng,min(16,len(episodes)),12,device)
        fixed,adapted,z0,z,inner=meta_prediction(model,batch,False)
        target=batch[-1]
        values.append(dict(fixed_corner_rmse=float((fixed-target).square().mean().sqrt()),
                           adapted_corner_rmse=float((adapted-target).square().mean().sqrt()),
                           latent_delta=float((z-z0).norm(dim=-1).mean()),
                           gradient_mean=float(np.mean([x['gradient'] for x in inner]))))
        reference,first,support,query,A,C,target=batch
        with torch.no_grad():
            changed=query.clone();changed[:,:48]=0
            _,zero=model.encode(first,changed,reference)
            zero_points=model.decode(reference,zero,z0)
            permutation=query.clone();permutation[:,:48]=query[torch.arange(len(query)-1,-1,-1),:48]
            _,permuted=model.encode(first,permutation,reference)
            permuted_points=model.decode(reference,permuted,z0)
            dependence.append(dict(zero_current_dino_points_delta=float((fixed-zero_points).norm(dim=-1).mean()),
                 permuted_current_dino_points_delta=float((fixed-permuted_points).norm(dim=-1).mean()),
                 zero_current_dino_corner_rmse=float((zero_points-target).square().mean().sqrt()),
                 permuted_current_dino_corner_rmse=float((permuted_points-target).square().mean().sqrt())))
    return dict(query={k:float(np.mean([r[k] for r in values])) for k in values[0]},
                dino_dependency_diagnostic={k:float(np.mean([r[k] for r in dependence])) for k in dependence[0]})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True,choices=['cup','insert']);parser.add_argument('--train',required=True);parser.add_argument('--train-labels',required=True)
    parser.add_argument('--dev',required=True);parser.add_argument('--dev-labels',required=True);parser.add_argument('--out',required=True);parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--updates',type=int,default=1200);parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    sources=['prove/fast_adapter/neural_geometry_adapter.py','tools/fast_adapter/p43_train.py']
    inputs={key:dict(path=getattr(args,key),sha256=sha(getattr(args,key))) for key in ('train','train_labels','dev','dev_labels')}
    write(out/'FREEZE.json',dict(args=vars(args),sources={s:sha(s) for s in sources},inputs=inputs,
        training_labels_permission=True,online_truth_permission=False,checkpoint_rule='final update',time=time.time()))
    train=load_data(args.train,args.train_labels,args.task);dev=load_data(args.dev,args.dev_labels,args.task)
    if len(train)<32 or len(dev)<4:raise RuntimeError(f'Insufficient visible training/development episodes: {len(train)}, {len(dev)}')
    model=NeuralGeometry(args.task).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    rng=np.random.default_rng(args.seed);log=[];start=time.time()
    for step in range(args.updates):
        model.train();S=int(rng.choice([3,6,12]));batch=batch_items(train,rng,args.batch_size,S,args.device)
        fixed,adapted,z0,z,inner=meta_prediction(model,batch,True)
        target=batch[-1];fixed_loss=((fixed-target)/GEOMETRY_SCALE).square().mean();adapted_loss=((adapted-target)/GEOMETRY_SCALE).square().mean()
        loss=.5*(fixed_loss+adapted_loss)+1e-4*z0.square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
        if step%50==0 or step+1==args.updates:
            row=dict(step=step+1,S=S,loss=float(loss.detach()),fixed_rmse=float(fixed_loss.detach().sqrt()*GEOMETRY_SCALE),
                     adapted_rmse=float(adapted_loss.detach().sqrt()*GEOMETRY_SCALE),z_change=float((z-z0).norm(dim=-1).mean().detach()),
                     inner_gradient=float(np.mean([r['gradient'] for r in inner])),elapsed=time.time()-start)
            log.append(row);print(json.dumps(row),flush=True);write(out/'LOG.json',log)
    torch.save(dict(model_config=model.config(),model=model.state_dict(),seed=args.seed,updates=args.updates,
                    data_inputs=inputs,sources={s:sha(s) for s in sources}),out/'checkpoint.pt')
    diagnostic=evaluation(model,dev,args.device);diagnostic.update(train_episodes=len(train),development_episodes=len(dev),
        checkpoint_sha256=sha(out/'checkpoint.pt'),elapsed=time.time()-start,checkpoint_selection='final update')
    write(out/'DIAGNOSTIC.json',diagnostic);print(json.dumps(diagnostic),flush=True)

if __name__=='__main__':main()
