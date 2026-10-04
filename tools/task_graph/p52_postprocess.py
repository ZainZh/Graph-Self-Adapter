#!/usr/bin/env python3
"""Explicit postcollection phases: prepare, features (3 GPUs), train (2 GPUs).

Nothing runs automatically. Failed conditions remain in the 24-row table. No
seed replacement or result-dependent split. Physical evaluation is a later freeze.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from tools.task_graph.p52_pack_dataset import sha,lines,verify_trajectory,pack
from tools.task_graph.p52_deployment import source_closure

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def prepare(binding_path,queue_roots,output,visual_python,training_python,dino,lila,joint_config,bc_config):
    binding=json.loads(binding_path.read_text());conditions=binding['conditions']
    if len(conditions)!=24 or [c['index'] for c in conditions]!=list(range(24)):
        raise ValueError('Require exactly the frozen 24 ordered conditions')
    splits=[c['split'] for c in conditions]
    if splits.count('train')!=18 or splits.count('validation')!=6:raise ValueError('Frozen split must be 18 train / 6 validation')
    discovered={}
    for queue in queue_roots:
        for attempt_path in queue.glob('worker_*/job_*/ATTEMPT_RESULT.json'):
            attempt=json.loads(attempt_path.read_text());job=attempt['job']['id']
            if job in discovered:raise ValueError('Duplicate attempt; retries not permitted')
            discovered[job]=(attempt_path.parent/'episode',attempt_path)
    if set(discovered)!={f'job_{i:02d}' for i in range(24)}:
        raise ValueError('Wait until all 24 attempts have completed; no partial-cohort pipeline')
    output.mkdir(parents=True,exist_ok=False);(output/'features').mkdir();(output/'logs').mkdir()
    rows=[];jobs=[];episodes=[];entries=[]
    for condition in conditions:
        i=condition['index'];job=f'job_{i:02d}';root,attempt_path=discovered[job]
        attempt=json.loads(attempt_path.read_text())
        if attempt['job']['seed']!=condition['condition_seed']:raise ValueError('Attempt seed does not match frozen condition')
        result=json.loads((root/'RESULT.json').read_text()) if (root/'RESULT.json').exists() else {}
        if result and (result.get('job_id')!=job or result.get('split')!=condition['split']):
            raise ValueError('Result job/split differs from predeclared condition')
        frames=sorted((root/'public').glob('frame_*.npz')) if (root/'public').exists() else []
        sealed=False;reason=None;seal=None
        try:seal,_=verify_trajectory(root,require_local=False);sealed=True
        except (ValueError,KeyError,OSError) as e:reason=str(e)
        qualified=bool(result.get('teacher_qualified',False))
        transitions=len(lines(root/'public/TRANSITIONS.jsonl')) if (root/'public/TRANSITIONS.jsonl').exists() else 0
        eligible=condition['split']=='train' and sealed and qualified and transitions>=16
        initial=root/'private/frame_0000.npz'
        entry_available=sealed and result.get('prefix_qualified',False) and initial.exists()
        if entry_available:
            with np.load(initial,allow_pickle=False) as f:
                entry_available=all(k in f for k in ('supervisor_sim_state','supervisor_ctrl'))
        row=dict(index=i,job_id=job,split=condition['split'],condition=condition,directory=str(root.resolve()),
                 attempt_sha256=sha(attempt_path),local_frames=len(frames),transitions=transitions,sealed=sealed,
                 teacher_qualified=qualified,eligible_training=eligible,entry_available=bool(entry_available),
                 result=result,seal_error=reason)
        rows.append(row)
        if frames:
            cache=(output/'features'/f'{job}.pt').resolve()
            jobs.append(dict(index=i,gpu=i%3,directory=str(root.resolve()),frames=[str(p.resolve()) for p in frames],
                             frame_sha256=[sha(p) for p in frames],output=str(cache)))
            if eligible or (condition['split']=='validation' and sealed and transitions>=2):
                episode_id=lines(root/'public/TRANSITIONS.jsonl')[0]['episode_id']
                episodes.append(dict(episode_id=episode_id,split=condition['split'],directory=str(root.resolve()),feature_cache=str(cache)))
        if condition['split']=='validation' and entry_available:
            entries.append(dict(name=job,source_kind='p52_real_physical_predecessor',split='validation',
                                frame=str(initial.resolve()),frame_sha256=sha(initial),condition=condition,
                                source_seal=str((root/'SEAL.json').resolve()),source_seal_sha256=sha(root/'SEAL.json')))
    valid_train=sum(r['eligible_training'] for r in rows);ready=valid_train>=12 and len(entries)==6
    plan=dict(schema='p52.postcollection.v1',binding=dict(path=str(binding_path.resolve()),sha256=sha(binding_path)),
              rows=rows,feature_jobs=jobs,episodes=episodes,validation_entries=entries,
              gate=dict(ready=ready,qualified_train=valid_train,required_train=12,validation_entries=len(entries),required_validation=6),
              visual_python=str(visual_python),training_python=str(training_python),dino=str(dino),lila=str(lila),
              configs={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in [('joint',joint_config),('bc',bc_config)]},
              sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/p52_postprocess.py','tools/task_graph/p52_train.py','prove/task_graph/p52_visual_features.py'])},
              visual_assets={str(p.resolve()):sha(p) for p in [dino/'model.safetensors',dino/'config.json',dino/'preprocessor_config.json',lila/'VISUAL_WEIGHTS.pt',lila/'models/vla_model_fm.py']},
              rule='All existing local trajectories receive features; only qualified >=16 train trajectories train. No replacements. Physical evaluation separately frozen.')
    write(output/'POSTPROCESS_PLAN.json',plan);write(output/'ALL_24_RESULTS.json',rows)
    write(output/'DATASET_MANIFEST.json',dict(episodes=episodes));write(output/'VALIDATION_ENTRIES.json',entries)
    return plan


def load_plan(path):
    plan=json.loads(path.read_text())
    if plan['schema']!='p52.postcollection.v1' or sha(plan['binding']['path'])!=plan['binding']['sha256']:
        raise ValueError('Postcollection binding changed')
    for source,digest in plan['sources'].items():
        if sha(ROOT/source)!=digest:raise ValueError('Frozen training/feature source changed')
    for source,digest in plan['visual_assets'].items():
        if sha(source)!=digest:raise ValueError('Frozen pretrained frontend asset changed')
    for item in plan['configs'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Training configuration changed')
    return plan


def feature_worker(plan_path,gpu):
    from prove.task_graph.p52_visual_features import FrozenStageVision,public_features
    plan=load_plan(plan_path);torch.set_num_threads(1)
    vision=FrozenStageVision(plan['dino'],plan['lila'],'cuda')
    for job in plan['feature_jobs']:
        if job['gpu']!=gpu:continue
        target=Path(job['output'])
        if target.exists():raise FileExistsError(target)
        rows=[];sources=[]
        for path,digest in zip(job['frames'],job['frame_sha256']):
            if sha(path)!=digest:raise ValueError('Public frame changed after pipeline prepare')
            with np.load(path,allow_pickle=False) as frame:rows.append(public_features(frame,vision))
            sources.append(dict(path=path,sha256=digest))
        torch.save(dict(schema='p52_public_features_v01',features={k:torch.as_tensor(np.stack([r[k] for r in rows])) for k in rows[0]},
                        sources=sources,frontend=vision.binding),target)
        print(json.dumps(dict(job=job['index'],frames=len(rows),output=str(target))),flush=True)


def launch(plan_path,phase):
    plan=load_plan(plan_path);folder=plan_path.parent;commands=[]
    if phase=='features':
        for gpu in range(3):commands.append((gpu,[plan['visual_python'],'-m','tools.task_graph.p52_postprocess','worker','--plan',str(plan_path.resolve()),'--plan-sha256',sha(plan_path),'--gpu',str(gpu)],f'features_gpu{gpu}'))
    else:
        if not plan['gate']['ready']:raise ValueError('Need >=12 qualified training trajectories and all six validation entries; root must reconsider, no automatic refill')
        receipt=json.loads((folder/'FEATURES_EXECUTION.json').read_text())
        if len(receipt)!=3 or any(r['returncode']!=0 for r in receipt) or any(not Path(j['output']).exists() for j in plan['feature_jobs']):
            raise ValueError('All three feature workers and every local-trajectory cache must complete first')
        # Re-read and verify actual seals/files immediately before packing/training.
        data=pack(dict(episodes=plan['episodes']));data_path=folder/'native_data.pt'
        if data_path.exists():raise FileExistsError(data_path)
        torch.save(data,data_path)
        for gpu,name in enumerate(('joint','bc')):
            commands.append((gpu,[plan['training_python'],'-m','tools.task_graph.p52_train','--data',str(data_path.resolve()),
                        '--config',plan['configs'][name]['path'],'--output',str((folder/f'{name}_final.pt').resolve()),'--device','cuda'],name))
    def execute(item):
        gpu,argv,name=item;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (folder/'logs'/f'{name}.log').open('x') as log:
            run=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        return dict(name=name,gpu=gpu,argv=argv,returncode=run.returncode,automatic_retry=False)
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:results=list(pool.map(execute,commands))
    write(folder/f'{phase.upper()}_EXECUTION.json',results)
    if any(r['returncode']!=0 for r in results):raise RuntimeError('A worker failed; retained, not retried')


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True)
    q=sub.add_parser('prepare');q.add_argument('--binding',type=Path,required=True);q.add_argument('--queue-roots',type=Path,nargs='+',required=True)
    for name in ('output','visual-python','training-python','dino','lila','joint-config','bc-config'):q.add_argument('--'+name,type=Path,required=True)
    for phase in ('features','train','worker'):
        q=sub.add_parser(phase);q.add_argument('--plan',type=Path,required=True);q.add_argument('--plan-sha256',required=True)
        if phase=='worker':q.add_argument('--gpu',type=int,choices=[0,1,2],required=True)
    a=p.parse_args()
    if a.phase=='prepare':
        plan=prepare(a.binding,a.queue_roots,a.output,a.visual_python,a.training_python,a.dino,a.lila,a.joint_config,a.bc_config);print(json.dumps(plan['gate']))
    else:
        if sha(a.plan)!=a.plan_sha256:raise ValueError('Postcollection plan SHA differs from execution receipt')
        if a.phase=='worker':feature_worker(a.plan,a.gpu)
        else:launch(a.plan,a.phase)


if __name__=='__main__':main()
