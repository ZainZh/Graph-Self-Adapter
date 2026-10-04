#!/usr/bin/env python3
"""Wait at most one hour, validate final training, run exactly the local pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def validate_training(a,root):
    import torch
    if sha(a.postprocess_plan)!=a.plan_sha256:raise ValueError('Postprocess plan changed')
    plan=read(a.postprocess_plan)
    if plan['schema']!='p52.postcollection.v1':raise ValueError('Wrong training plan')
    for path,digest in plan['sources'].items():
        if sha(root/path)!=digest:raise ValueError('Frozen training source changed: '+path)
    for path,digest in plan['visual_assets'].items():
        if sha(path)!=digest:raise ValueError('Frozen visual asset changed: '+path)
    if sha(a.environment_binding)!=plan['binding']['sha256']:raise ValueError('Collection binding changed')
    if a.environment_binding.resolve()!=Path(plan['binding']['path']).resolve():raise ValueError('Different collection binding path')
    if Path(plan['dino']).resolve()!=a.dino.resolve() or Path(plan['lila']).resolve()!=a.lila.resolve():
        raise ValueError('Different visual asset roots')
    receipt=read(a.postprocess_plan.parent/'TRAIN_EXECUTION.json')
    if len(receipt)!=2 or {r['name'] for r in receipt}!={'joint','bc'} or any(r['returncode']!=0 for r in receipt):
        raise ValueError('Both training workers must finish successfully')
    data_sha=sha(a.data);models={}
    if a.data.resolve()!=a.postprocess_plan.parent/'native_data.pt':raise ValueError('Wrong training dataset')
    if a.entries.resolve()!=a.postprocess_plan.parent/'VALIDATION_ENTRIES.json':raise ValueError('Wrong validation entries')
    for name in ('joint','bc'):
        config=getattr(a,name+'_config');checkpoint=getattr(a,name)
        if config.resolve()!=Path(plan['configs'][name]['path']).resolve() or sha(config)!=plan['configs'][name]['sha256']:
            raise ValueError('Frozen '+name+' configuration changed')
        if checkpoint.resolve()!=a.postprocess_plan.parent/(name+'_final.pt'):raise ValueError('Final checkpoints only')
        model=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if model['schema']!='p52_stage_skill_v01' or model['settings']!=read(config) or model['data_sha256']!=data_sha:
            raise ValueError(name+' checkpoint/settings/data mismatch')
        if not model['model'] or any(not torch.isfinite(t).all().item() for t in model['model'].values()):
            raise ValueError(name+' weights not finite')
        models[name]=dict(path=str(checkpoint),sha256=sha(checkpoint),config_sha256=sha(config),data_sha256=data_sha)
    entries=read(a.entries)
    if entries!=plan['validation_entries'] or len(entries)!=6 or any(e['split']!='validation' for e in entries):raise ValueError('Exact frozen six validation entries required')
    if not plan['gate']['ready']:raise ValueError('Frozen training gate not ready')
    if a.vision_python.resolve()!=Path(plan['visual_python']).resolve():raise ValueError('Different frozen visual interpreter')
    return dict(plan_sha256=a.plan_sha256,receipt_sha256=sha(a.postprocess_plan.parent/'TRAIN_EXECUTION.json'),
                models=models,entries_sha256=sha(a.entries))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('postprocess-plan','environment-binding','entries','joint','bc','data','joint-config','bc-config','dino','lila','vision-python','python','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--queue-sha256',required=True)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.absolute())
    root=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=False)
    try:
        queue=root/'tools/task_graph/p52_local_eval_queue.py'
        if sha(queue)!=a.queue_sha256:raise ValueError('Reviewed queue source changed')
        if sha(a.postprocess_plan)!=a.plan_sha256:raise ValueError('Reviewed postprocess plan changed')
        receipt=a.postprocess_plan.parent/'TRAIN_EXECUTION.json';deadline=time.monotonic()+3600
        while not receipt.is_file():
            if time.monotonic()>=deadline:raise TimeoutError('Training receipt absent after 3600 seconds')
            time.sleep(min(10,max(0,deadline-time.monotonic())))
        validated=validate_training(a,root);write(a.output/'TRAINING_ACCEPTANCE.json',validated)
        freeze=a.output/'POLICY_FREEZE.json'
        command=[str(a.python),'-m','tools.task_graph.p52_policy_pilot','prepare',
            '--environment-binding',str(a.environment_binding),'--entries',str(a.entries),'--bc',str(a.bc),
            '--joint',str(a.joint),'--dino',str(a.dino),'--lila',str(a.lila),
            '--vision-python',str(a.vision_python),'--vision-device','cuda','--output',str(freeze)]
        with (a.output/'prepare.log').open('x') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        freeze_sha=sha(freeze);write(a.output/'POLICY_FREEZE_SHA.json',dict(path=str(freeze),sha256=freeze_sha))
        if sha(queue)!=a.queue_sha256:raise ValueError('Reviewed queue source changed before launch')
        command=[str(a.python),'-m','tools.task_graph.p52_local_eval_queue','--binding',str(freeze),
                 '--binding-sha256',freeze_sha,'--python',str(a.python),'--output',str(a.output/'local_eval')]
        with (a.output/'queue.log').open('x') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        results=read(a.output/'local_eval/QUEUE_RESULT.json')['cells']
        if len(results)!=18:raise ValueError('Missing evaluation cells')
        summary=[]
        for row in results:
            item=dict(cell=row['cell'],execution_status=row['execution_status'],returncode=row['returncode'])
            result=row['artifacts'].get('RESULT.json')
            if result:
                if sha(result['path'])!=result['sha256']:raise ValueError('Result artifact changed')
                item['local_success']=read(result['path'])['local_success']
            summary.append(item)
        write(a.output/'PIPELINE_SUMMARY.json',dict(cells=summary,policy_freeze_sha256=freeze_sha,full_task_started=False,automatic_retries=0))
    except Exception as error:
        write(a.output/'PIPELINE_ABORT.json',dict(error=type(error).__name__,reason=str(error),traceback=traceback.format_exc(),automatic_retries=0))
        raise


if __name__=='__main__':main()
