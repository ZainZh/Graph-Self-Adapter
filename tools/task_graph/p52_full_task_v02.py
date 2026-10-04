#!/usr/bin/env python3
"""Three single original-task engineering attempts; not independent success rates."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import torch
from tools.task_graph import p52_full_task as original
from tools.task_graph.p52_full_task import (ROOT, InsertJudge, initialize, run_prefix, run_insert,
    StagePolicy, Config, StageRuntime, Vision, PublicRenderer, verify_collection, dump, sha, source_closure)

CELLS=[dict(index=i,arm=arm) for i,arm in enumerate(('bc','fixed','online'))]


def prepare(collection_binding,bc,joint,dino,lila,vision_python,vision_device):
    binding=original.prepare(collection_binding,joint,dino,lila,vision_python,vision_device)
    checkpoints={}
    datasets=set()
    for arm,path in (('bc',bc),('fixed',joint),('online',joint)):
        package=torch.load(path,map_location='cpu',weights_only=True)
        if package['schema']!='p52_stage_skill_v01' or bool(package['settings']['training'].get('ordinary',False))!=(arm=='bc'):
            raise ValueError('BC/joint checkpoint training type mismatch')
        model=StagePolicy(Config(**package['config']));model.load_state_dict(package['model'],strict=True)
        if any(not torch.isfinite(v).all().item() for v in model.state_dict().values()):raise ValueError('Nonfinite weights')
        datasets.add(package['data_sha256'])
        checkpoints[arm]=dict(path=str(path.resolve()),sha256=sha(path))
    if len(datasets)!=1:raise ValueError('BC/joint training data mismatch')
    binding.pop('checkpoint')
    binding.update(schema='p52.full-original-task.v2',cells=CELLS,checkpoints=checkpoints,
       sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/p52_full_task_v02.py','tools/task_graph/p52_full_task.py','tools/task_graph/p52_vision_worker.py'])},
       local_evidence='BC 6/6, joint fixed 1/6, online 1/6 on opened local validation; joint model not qualified or promoted',
       scope='Three single original-initial-state engineering attempts; BC flow ability and joint fixed/online attribution separate; no independent success-rate claim')
    return binding


def verify(binding):
    if binding['schema']!='p52.full-original-task.v2' or binding['cells']!=CELLS:raise ValueError('Wrong full-task scope')
    if set(binding['checkpoints'])!={'bc','fixed','online'}:raise ValueError('Missing checkpoint arm')
    if binding['automatic_retries']!=0 or binding['independent_statistical_evidence'] is not False:raise ValueError('Engineering-only no-retry scope required')
    required=set(source_closure(ROOT,['tools/task_graph/p52_full_task_v02.py','tools/task_graph/p52_full_task.py','tools/task_graph/p52_vision_worker.py']))
    if set(binding['sources'])!=required:raise ValueError('Incomplete full-task source closure')
    for key,value in dict(prefix_max_actions=400,insert_max_actions=180,original_initial_restores=1,later_state_restores=0,entry_transition_actions=0).items():
        if binding[key]!=value:raise ValueError('Full-task budget/init changed')
    if binding['grasp_offset']!=[0,0,0] or binding['entry_offset']!=[0,0,0]:raise ValueError('Perturbations prohibited')
    if sha(binding['parent']['path'])!=binding['parent']['sha256']:raise ValueError('Parent changed')
    parent=json.loads(Path(binding['parent']['path']).read_text());verify_collection(parent)
    if binding['environment']!=parent['environment'] or binding['prefix']!=parent['prefix'] or binding['public']!=parent['public']:raise ValueError('Frozen original environment/prefix differs')
    for name,digest in binding['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    for name,digest in binding['visual_assets'].items():
        if sha(name)!=digest:raise ValueError(f'Visual asset changed: {name}')
    for checkpoint in binding['checkpoints'].values():
        if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Stage model changed')
    if binding['checkpoints']['fixed']!=binding['checkpoints']['online']:raise ValueError('Fixed/online checkpoint mismatch')
    for name,version in binding['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Runtime package changed')


def run(binding_path,index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if index not in (0,1,2):raise ValueError('Only three original-state engineering cells are authorized')
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('prefix','public','private'):(output/folder).mkdir()
    dump(output/'BINDING.json',binding);cell=binding['cells'][index];dump(output/'CELL.json',cell)
    torch.set_num_threads(1);torch.manual_seed(binding['seed']);np.random.seed(binding['seed'])
    ck=torch.load(binding['checkpoints'][cell['arm']]['path'],map_location='cpu',weights_only=True)
    model=StagePolicy(Config(**ck['config']));model.load_state_dict(ck['model'],strict=True);runtime=StageRuntime(model,online=cell['arm']=='online')
    before={k:v.clone() for k,v in model.state_dict().items()};env=renderer=vision=None
    try:
        env=initialize(binding,output)
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('geom','ngeom')]}
        dump(output/'private/NAMES.json',names)
        prefix=run_prefix(env,binding,output/'prefix',output.name);dump(output/'PREFIX_RESULT.json',prefix)
        if prefix['qualified']:
            vision=Vision(binding['vision']);dump(output/'VISUAL_BINDING.json',vision.binding)
            renderer=PublicRenderer(env,binding['public'])
            result=run_insert(env,binding,runtime,vision,renderer,output,names)
        else:result=dict(actions=0,graph_complete=False,official_success=bool(env._check_success()),stop_reason=prefix['stop_reason'],updates=0)
        if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise RuntimeError('Stage network mutated')
        verify(binding)
        files={str(p.relative_to(output)):sha(p) for folder in ('prefix','public','private') for p in sorted((output/folder).iterdir())}
        files['PREFIX_RESULT.json']=sha(output/'PREFIX_RESULT.json');files['CELL.json']=sha(output/'CELL.json')
        dump(output/'SEAL.json',dict(binding_sha256=sha(output/'BINDING.json'),files=files,cell=cell,original_restores=1,later_restores=0))
        result.update(cell=cell,prefix_actions=prefix['actions'],total_actions=prefix['actions']+result['actions'],
                      seal_sha256=sha(output/'SEAL.json'),scope='complete original task development engineering; not independent statistical evidence',automatic_retry=False)
        dump(output/'RESULT.json',result);return result
    except Exception as e:
        dump(output/'ABORT.json',dict(error=type(e).__name__,reason=str(e),cell=cell,automatic_retry=False));raise
    finally:
        runtime.exit()
        if renderer is not None:renderer.close()
        if env is not None:env.close()
        if vision is not None:vision.close()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('collection-binding','bc','joint','dino','lila','output'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--vision-python',required=True);q.add_argument('--vision-device',default='cuda')
    q=sub.add_parser('run');q.add_argument('--binding',type=Path,required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        binding=prepare(a.collection_binding,a.bc,a.joint,a.dino,a.lila,a.vision_python,a.vision_device)
        verify(binding);a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,binding)
    else:print(json.dumps(run(a.binding,a.index,a.output)))


if __name__=='__main__':main()
