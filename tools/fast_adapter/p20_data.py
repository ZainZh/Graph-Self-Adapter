"""Real MuJoCo contact transitions + frozen DINO ROI features for P20."""
from __future__ import annotations
import argparse,json,time,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import torch
from prove.fast_adapter.insert3d import *
from prove.fast_adapter.visual_alignment import FrozenDinoROI,DEFAULT_MODEL
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p20_insert3d_v01')

def query_samples(seed):
 rng=np.random.default_rng(seed)
 return [(np.r_[rng.uniform(-.04,.04,2),rng.uniform(.13,.16)],0),
         (np.r_[rng.uniform(-.012,.012,2),rng.uniform(.105,.13)],10),
         (np.r_[rng.uniform(-.008,.008,2),rng.uniform(.103,.115)],18),
         (np.r_[rng.uniform(-.002,.002,2),rng.uniform(.044,.064)],32)]

def collect(shard,n=256,model_dir=DEFAULT_MODEL,device='cuda'):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 out=R/'bank'/f'shard{shard}';out.mkdir(parents=True,exist_ok=False)
 env=Insert3D();enc=FrozenDinoROI(model_dir,device,microbatch=16)
 support=[];query=[];query_next=[];bases=[];teachers=[];query_action=[];states=[];physics=[];visibility=[];began=time.time()
 pending=[];pending_boxes=[];encoded=[]
 def append_obs():
  rgb,box,count=env.observe();pending.append(rgb);pending_boxes.append(box);visibility.append(count)
 def encode():
  ft=enc.encode(np.stack(pending),np.stack(pending_boxes)).cpu().half()
  encoded.extend(ft.unbind());pending.clear();pending_boxes.clear()
 for i in range(n):
  sid=shard*n+i;M,b,fr=domain(2026092400+sid);env.set_domain(M,b,fr)
  env.reset(start_position(2027192400+sid));append_obs()
  for a in PROBES:env.step(a);append_obs()
  bs=[];ts=[];qs=[];st=[]
  rng=np.random.default_rng(2028092400+sid)
  for pos,t in query_samples(2029092400+sid):
   env.reset(pos);append_obs();base=nominal_action(pos,t);target=oracle_action(pos,t,M,b)
   action=np.clip(target+rng.uniform(-.2,.2,3),-1,1)
   env.step(action);append_obs();bs.append(base);ts.append(target);qs.append(action);st.append(pos)
  bases.append(bs);teachers.append(ts);query_action.append(qs);states.append(st);physics.append({'seed':2026092400+sid,'M':M.tolist(),'b':b.tolist(),'friction':fr})
  if len(pending)>=112:encode()
  if (i+1)%64==0:print(json.dumps({'shard':shard,'domains':i+1,'seconds':time.time()-began}),flush=True)
 if pending:encode()
 feats=torch.stack(encoded).reshape(n,14,2,1024)
 bank={'support':feats[:,:6],'query':feats[:,6::2],'query_next':feats[:,7::2],'base':torch.tensor(np.array(bases),dtype=torch.float32),'teacher':torch.tensor(np.array(teachers),dtype=torch.float32),'query_action':torch.tensor(np.array(query_action),dtype=torch.float32),'states_diagnostic_only':torch.tensor(np.array(states),dtype=torch.float32),'physics_evaluator_only':physics}
 torch.save(bank,out/'BANK.pt');save_json(out/'RECEIPT.json',{'shard':shard,'domains':n,'images':n*14,'seconds':time.time()-began,'min_visible_pixels':np.array(visibility).min(0).tolist(),'sha256':digest(out/'BANK.pt'),'actual_mujoco_rgb':True,'coords_not_runtime_input':True})
 env.close();print('BANK_COMPLETE',shard,flush=True)

def merge():
 banks=[torch.load(R/'bank'/f'shard{i}'/'BANK.pt',weights_only=False) for i in range(4)]
 full={k:torch.cat([b[k] for b in banks]) for k in banks[0] if isinstance(banks[0][k],torch.Tensor)}
 full['physics_evaluator_only']=sum([b['physics_evaluator_only'] for b in banks],[])
 torch.save(full,R/'bank'/'ALL.pt')
 # Representation fitted on train domains0..1023. No final/dev trajectories.
 feat=torch.cat([full['support'].flatten(0,1),full['query'].flatten(0,1),full['query_next'].flatten(0,1)])
 pcas=[]
 for obj in range(2):
  x=feat[:,obj].float().cuda();mu=x.mean(0);x=x-mu;cov=x.T@x/(len(x)-1);values,vectors=torch.linalg.eigh(cov);v=vectors[:,-32:];sd=values[-32:].clamp_min(1e-6).sqrt()
  pcas.append({'mean':mu.cpu(),'basis':v.cpu(),'sd':sd.cpu(),'retained_variance':float(values[-32:].sum()/values.sum())})
 torch.save(pcas,R/'bank'/'PCA.pt')
 def transform(f):return torch.cat([((f[...,i,:].float()-pcas[i]['mean'])@pcas[i]['basis']/pcas[i]['sd']) for i in range(2)],-1)
 z={k:transform(full[k]) for k in ['support','query','query_next']}
 z.update({k:full[k] for k in ['base','teacher','query_action','states_diagnostic_only']})
 torch.save(z,R/'bank'/'TRAIN.pt')
 save_json(R/'bank'/'MERGE.json',{'domains':len(full['base']),'images':len(feat),'pca_dimensions':64,'retained_variance':[p['retained_variance'] for p in pcas],'pca_labels':'none, unsupervised train RGB only','all_sha256':digest(R/'bank'/'ALL.pt'),'train_sha256':digest(R/'bank'/'TRAIN.pt'),'pca_sha256':digest(R/'bank'/'PCA.pt')})
 print('MERGED',len(full['base']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','merge']);p.add_argument('--shard',type=int);p.add_argument('--domains',type=int,default=256);p.add_argument('--dino',default=DEFAULT_MODEL);a=p.parse_args()
 if a.mode=='collect':collect(a.shard,a.domains,a.dino)
 else:merge()
