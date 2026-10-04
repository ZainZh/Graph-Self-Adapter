"""Finite P65 freeze/queue: first formal cell alone, then at most two workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from prove.fast_adapter.p61_data import sha256
from tools.fast_adapter.p64_cup_physical import source_closure
from tools.fast_adapter.p65_cup_evaluate import cells,validate

ROOT=Path(__file__).resolve().parents[2]
QUEUE=dict(first_index_alone=0,max_workers_after_first=2,cpu_threads=1,automatic_retries=0,
    on_program_error='stop_new_submissions_and_await_running')


def prepare(output,plan,*,draft=True):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    p64_path=ROOT/'evaluations/fast_adapter_p64_task_transfer_v01/cup/physical_v01/PHYSICAL_FREEZE.json'
    p64=json.loads(p64_path.read_text())
    sources=source_closure(['prove/fast_adapter/p65_cup.py','tools/fast_adapter/p65_cup_evaluate.py',
        'tools/fast_adapter/p65_cup_physical.py','tests/test_p65_cup.py'])
    files={str(path.relative_to(ROOT)):sha256(path) for path in sources}
    # Runtime non-Python assets are identical to P64. Historical presentation
    # code is deliberately not part of this runtime dependency closure.
    for name,digest in p64['files'].items():
        if not name.endswith('.py'):
            if sha256(name)!=digest:raise ValueError('Historical bound asset changed: '+name)
            files[name]=digest
    official='data/sources/lila-wam-reference-20260923/models/vla_model_fm.py'
    files[official]=sha256(official)
    missing=[]
    if Path(plan).is_file():files[str(plan)]=sha256(plan)
    else:missing.append(str(plan))
    qualification=Path(plan).parent/'ROUTING_QUALIFICATION.json'
    if qualification.is_file():files[str(qualification)]=sha256(qualification)
    else:missing.append(str(qualification))
    if missing and not draft:raise FileNotFoundError('Missing root plan')
    value=dict(schema='prove.p65-cup-local-routing-freeze.v1',status='draft' if draft else 'frozen',
        result_state='before_P65_physics',steps=240,cells=cells(),queue=QUEUE,
        plan=str(plan),encoder=p64['encoder'],action_policy=p64['action_policy'],dino_model=p64['dino_model'],
        files=files,missing_assets=missing,source_closure_count=len(sources),
        upstream_P64_freeze_sha256=sha256(p64_path),condition_identity='opened P64 development conditions; not fresh held-out',
        routing=dict(neural='phase_before=approach and finite visible geometry only',
            code='observe, clearance, align, descend, close, transport, hold',
            neural_phase_input='approach',gates='original unmodified tolerances and three stable samples',
            invalid_geometry='original code hold; no neural call'),
        updates=dict(fixed='never',online_history='all preclose observations as P64',
            online_local='approach only; z reset at entry; retained causal public history and fixed first-EEF anchor',
            exit='clear live z after critical exit; preserve archived predictions and frozen decoded target; budget-end reset if no exit'),
        success=p64['success'],mechanical=p64['mechanical'],runtime_inputs=p64['runtime_inputs'],
        orientation=p64['orientation'],selected_videos=[dict(split=s,seed=640100) for s in ('id','size','position')])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n')
    return dict(path=str(output),sha256=sha256(output),source_closure_count=len(sources),missing_assets=missing)


def queue(freeze,output,device='cuda'):
    contract=validate(freeze);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pending=list(enumerate(contract['cells']));active={};records=[];failed=False;first_completed=False
    environment=dict(os.environ,MUJOCO_GL='egl',EGL_DEVICE_ID='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def launch(index,cell):
        destination=output/f"{index:02d}_{cell['split']}_{cell['seed']}_{cell['arm']}"
        command=[sys.executable,'-m','tools.fast_adapter.p65_cup_evaluate','--freeze',str(freeze),
            '--index',str(index),'--output',str(destination),'--device',device]
        log=(output/f'{index:02d}.log').open('w')
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        active[index]=dict(process=process,log=log,started=time.monotonic(),cell=cell,output=str(destination))
    # Queue identity and total budget are fixed before launching any child.
    (output/'QUEUE_LAUNCH.json').write_text(json.dumps(dict(freeze_sha256=sha256(freeze),queue=QUEUE,
        cells=contract['cells'],maximum_attempts=36),indent=2)+'\n')
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
            record=dict(index=index,cell=job['cell'],exit_code=code,output=job['output'],
                elapsed_seconds=time.monotonic()-job['started'],automatic_retries=0)
            records.append(record);print(json.dumps(record),flush=True)
            if code:failed=True
            elif index==0:first_completed=True
        for index in finished:del active[index]
        (output/'QUEUE_PROGRESS.json').write_text(json.dumps(dict(completed=records,running=list(active),
            pending=[i for i,_ in pending],program_error_seen=failed),indent=2)+'\n')
        if active:time.sleep(.5)
    receipt=dict(completed=len(records),records=records,unstarted=[i for i,_ in pending],
        maximum_attempts=36,automatic_retries=0,program_error_seen=failed)
    (output/'QUEUE_COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if failed:raise RuntimeError('P65 child program error; no new submissions; running children awaited')


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    for name in ('draft','freeze'):
        a=s.add_parser(name);a.add_argument('--output',required=True);a.add_argument('--plan',required=True)
    q=s.add_parser('queue');q.add_argument('--freeze',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.command=='queue':queue(a.freeze,a.output,a.device)
    else:print(json.dumps(prepare(a.output,a.plan,draft=a.command=='draft'),indent=2))
