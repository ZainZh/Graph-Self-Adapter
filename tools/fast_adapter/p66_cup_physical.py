"""Explicit P66 freeze preparation and finite queue; never auto-launch."""
import os
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['MUJOCO_EGL_DEVICE_ID']='0'
os.environ.setdefault('EGL_DEVICE_ID','0')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from prove.fast_adapter.p61_data import sha256
from prove.fast_adapter.p66_cup import ARMS
from tools.fast_adapter.p64_cup_physical import source_closure
from tools.fast_adapter.p66_cup_evaluate import cells,validate,QUEUE

ROOT=Path(__file__).resolve().parents[2]


def prepare(output,plan,*,kind,draft=False):
    output,plan=Path(output),Path(plan)
    if output.exists():raise FileExistsError(output)
    roots=['prove/fast_adapter/p66_cup.py','tools/fast_adapter/p66_cup_physical.py','tests/test_p66_cup.py']
    roots.append('tools/fast_adapter/p66_cup_replay.py' if kind=='replay' else 'tools/fast_adapter/p66_cup_evaluate.py')
    closure=source_closure(roots);files={str(p.relative_to(ROOT)):sha256(p) for p in closure}
    missing=[]
    if plan.is_file():files[str(plan)]=sha256(plan)
    else:missing.append(str(plan))
    if kind=='replay':
        report=ROOT/'evaluations/fast_adapter_p65_local_routing_v01/report_v01/RESULTS.json'
        data=json.loads(report.read_text());files[str(report)]=sha256(report)
        sources=[]
        for item in data['cells']:
            cell=item['result']['cell']
            if cell['arm']!='online_local':continue
            path=Path(item['path']);seal=json.loads((path/'SEAL.json').read_text())
            files[str(path/'SEAL.json')]=sha256(path/'SEAL.json')
            for name,digest in seal['files'].items():
                if sha256(path/name)!=digest:raise ValueError('Source seal changed')
                files[str(path/name)]=digest
            sources.append(dict(cell=cell,path=str(path),seal_sha256=sha256(path/'SEAL.json')))
        if len(sources)!=12:raise ValueError('Expected12P65online_local trajectories')
        result=dict(schema='prove.p66-cup-replay-freeze.v1',status='draft' if draft else 'frozen',plan=str(plan),
            arms=list(ARMS),sources=sources,files=files,missing_assets=missing,
            observation_times=list(range(0,240,4)),predictions=48,physics_steps=0,training_updates=0,
            protocol='all48predictionbytessealed beforeinvariants; rawreadoutonlyatmatching savedtimestamps; nofreshpostclose raw',
            scope='sparse public engineering replay, not task success or dense equivalence')
    elif kind=='physical':
        upstream=ROOT/'evaluations/fast_adapter_p64_task_transfer_v01/cup/physical_v01/PHYSICAL_FREEZE.json'
        p64=json.loads(upstream.read_text())
        for name,digest in p64['files'].items():
            if not name.endswith('.py'):
                if sha256(name)!=digest:raise ValueError('Historical asset changed: '+name)
                files[name]=digest
        official='data/sources/lila-wam-reference-20260923/models/vla_model_fm.py';files[official]=sha256(official)
        qualification=plan.parent/'QUALIFICATION.json'
        if qualification.is_file():files[str(qualification)]=sha256(qualification)
        else:missing.append(str(qualification))
        result=dict(schema='prove.p66-cup-reference-freeze.v1',status='draft' if draft else 'frozen',
            result_state='beforeP66physics',plan=str(plan),qualification=str(qualification),steps=240,cells=cells(),queue=QUEUE,
            encoder=p64['encoder'],action_policy=p64['action_policy'],dino_model=p64['dino_model'],
            files=files,missing_assets=missing,source_closure_count=len(closure),
            condition_identity='same openedP64/P65development conditions; nofreshheldout claim',
            success=p64['success'],mechanical=p64['mechanical'],orientation=p64['orientation'],runtime_inputs=p64['runtime_inputs'],
            reference='latest admitted rawLiLa geometry8 atfirstapproach, originalfirstEEFanchor; neverrefresh lockedreference',
            entry='once firstreference+threecausalsupport available; currentadmissionnotrequired; deferredifmissing',
            continuous='afterentry, onlynewadmittedsupport; maxoneoriginal3step eventperframe',
            end='resetactivez; preservecached decodedtarget andarchives',selected_videos=[dict(split=s,seed=640100) for s in ('id','size','position')])
    else:raise ValueError('Unknown freeze kind')
    if missing and not draft:raise FileNotFoundError('Missing root artifacts: '+str(missing))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return dict(path=str(output),sha256=sha256(output),missing_assets=missing)


def queue(freeze,output,device='cuda'):
    contract=validate(freeze);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pending=list(enumerate(contract['cells']));active={};records=[];failed=False;first_completed=False
    environment=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',EGL_DEVICE_ID='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def launch(index,cell):
        destination=output/f"{index:02d}_{cell['split']}_{cell['seed']}_{cell['arm']}"
        command=[sys.executable,'-m','tools.fast_adapter.p66_cup_evaluate','--freeze',str(freeze),'--index',str(index),
            '--output',str(destination),'--device',device]
        log=(output/f'{index:02d}.log').open('w')
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        active[index]=dict(process=process,log=log,started=time.monotonic(),cell=cell,output=str(destination))
    (output/'QUEUE_LAUNCH.json').write_text(json.dumps(dict(freeze_sha256=sha256(freeze),queue=QUEUE,cells=contract['cells'],maximum_attempts=48),indent=2)+'\n')
    while pending or active:
        cap=2 if first_completed else 1
        while not failed and pending and len(active)<cap:
            if not first_completed and active:break
            index,cell=pending.pop(0);launch(index,cell)
            if not first_completed:break
        if not active:break
        finished=[]
        for index,job in active.items():
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();finished.append(index)
            record=dict(index=index,cell=job['cell'],exit_code=code,output=job['output'],elapsed_seconds=time.monotonic()-job['started'],automatic_retries=0)
            records.append(record);print(json.dumps(record),flush=True)
            if code:failed=True
            elif index==0:first_completed=True
        for index in finished:del active[index]
        (output/'QUEUE_PROGRESS.json').write_text(json.dumps(dict(completed=records,running=list(active),pending=[i for i,_ in pending],program_error_seen=failed),indent=2)+'\n')
        if active:time.sleep(.5)
    (output/'QUEUE_COMPLETION.json').write_text(json.dumps(dict(completed=len(records),records=records,unstarted=[i for i,_ in pending],
        maximum_attempts=48,automatic_retries=0,program_error_seen=failed),indent=2)+'\n')
    if failed:raise RuntimeError('P66programerror: new submissions stopped; running children awaited')


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    for name in ('replay-freeze','physical-freeze'):
        a=s.add_parser(name);a.add_argument('--output',required=True);a.add_argument('--plan',required=True);a.add_argument('--draft',action='store_true')
    q=s.add_parser('queue');q.add_argument('--freeze',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.command=='queue':queue(a.freeze,a.output,a.device)
    else:print(json.dumps(prepare(a.output,a.plan,kind=a.command.split('-')[0],draft=a.draft),indent=2))
