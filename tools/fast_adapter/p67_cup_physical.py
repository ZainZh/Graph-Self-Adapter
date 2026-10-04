"""P67 result-free freeze and bounded local queue; no outcome inspection."""
import os
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['MUJOCO_EGL_DEVICE_ID']='0'
os.environ.setdefault('EGL_DEVICE_ID','0')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from prove.fast_adapter.p61_data import sha256
from tools.fast_adapter.p64_cup_physical import source_closure
from tools.fast_adapter.p67_common import cells,selected_videos,QUEUE,FREEZE_SCHEMA,CELL_COUNT,CONDITION_COUNT
from tools.fast_adapter.p67_cup_evaluate import validate

ROOT=Path(__file__).resolve().parents[2]



def freshness_bindings(path):
    """Validate root qualification and bind direct evidence files, not inventory entries."""
    audit=json.loads(Path(path).read_text())
    if audit.get('passed') is not True:raise ValueError('Freshness audit did not pass')
    identities=[(c['split'],c['seed']) for c in audit.get('conditions',[])]
    expected={(c['split'],c['seed']) for c in cells()}
    if len(identities)!=CONDITION_COUNT or set(identities)!=expected:
        raise ValueError('Freshness audit condition identities differ from frozen cohort')
    bindings=audit.get('bindings')
    if not isinstance(bindings,list) or not bindings:raise ValueError('Freshness evidence bindings missing')
    files={}
    for binding in bindings:
        name=binding['path']; source=Path(name)
        if not source.is_absolute():source=ROOT/source
        if not source.is_file():raise FileNotFoundError(source)
        if sha256(source)!=binding['sha256']:raise ValueError('Freshness evidence changed: '+name)
        files[name]=binding['sha256']
    return files


def prepare(output,plan,*,draft=False):
    output,plan=Path(output),Path(plan)
    if output.exists():raise FileExistsError(output)
    upstream=ROOT/'evaluations/fast_adapter_p66_reference_state_v01/physical_v01/PHYSICAL_FREEZE.json'
    p66=json.loads(upstream.read_text())
    roots=['tools/fast_adapter/p67_common.py','tools/fast_adapter/p67_cup_evaluate.py',
        'tools/fast_adapter/p67_cup_physical.py','tests/test_p67_cup.py',
        'tools/fast_adapter/p67_report.py','tests/test_p67_report.py']
    closure=source_closure(roots);files={str(p.relative_to(ROOT)):sha256(p) for p in closure}
    # Original P66 non-code bindings own the unchanged checkpoints, physical
    # assets, DINO preprocessing and root's prior qualification.
    for name,digest in p66['files'].items():
        if not name.endswith('.py'):
            if sha256(name)!=digest:raise ValueError('P66 bound asset changed: '+name)
            files[name]=digest
    official='data/sources/lila-wam-reference-20260923/models/vla_model_fm.py';files[official]=sha256(official)
    freshness=plan.parent/'FRESHNESS_AUDIT.json'
    qualification=ROOT/'evaluations/fast_adapter_p66_reference_state_v01/QUALIFICATION.json'
    acceptance=ROOT/'evaluations/fast_adapter_p66_reference_state_v01/ROOT_FINAL_ACCEPTANCE.json'
    missing=[]
    for path in (plan,freshness,qualification,acceptance,upstream):
        if path.is_file():files[str(path)]=sha256(path)
        else:missing.append(str(path))
    if freshness.is_file():files.update(freshness_bindings(freshness))
    if missing and not draft:raise FileNotFoundError('Missing root readiness artifacts: '+str(missing))
    result=dict(schema=FREEZE_SCHEMA,status='draft' if draft else 'frozen',result_state='beforeP67physics_and_outcomes',
        plan=str(plan),freshness_audit=str(freshness),qualification=str(qualification),p66_acceptance=str(acceptance),
        upstream_p66_freeze=dict(path=str(upstream),sha256=sha256(upstream)),steps=240,conditions=CONDITION_COUNT,
        cells=cells(),queue=QUEUE,encoder=p66['encoder'],action_policy=p66['action_policy'],dino_model=p66['dino_model'],
        files=files,missing_assets=missing,source_closure_count=len(closure),
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','mujoco','transformers','scipy','Pillow')},
        condition_identity='fresh predeclared conditions verified by root freshness audit',
        runtime_identity='unchanged P66 state math, P65 routed controller, P64 trained models and original mechanical evaluator',
        runtime_inputs=p66['runtime_inputs'],success=p66['success'],mechanical=p66['mechanical'],orientation=p66['orientation'],
        selected_videos=selected_videos(),outcome_opening='only after all768cells sealed and completed; status reads filenames only',
        training_updates=0,automatic_retries=0)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return dict(path=str(output),sha256=sha256(output),missing_assets=missing)


def status(directory):
    """Only numeric file counts; never open a result, trace or child log."""
    root=Path(directory)
    return dict(expected=CELL_COUNT,started=len(list(root.glob('*/PRE_RUN.json'))),
        sealed=len(list(root.glob('*/SEAL.json'))),completed=len(list(root.glob('*/RESULT.json'))),
        program_errors=len(list(root.glob('*/PROGRAM_ERROR.json'))))


def queue(freeze,output,device='cuda'):
    contract=validate(freeze);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pending=list(enumerate(contract['cells']));active={};records=[];failed=False;first_completed=False
    environment=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',EGL_DEVICE_ID='0',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def launch(index,cell):
        destination=output/f"{index:03d}_{cell['split']}_{cell['seed']}_{cell['arm']}"
        command=[sys.executable,'-m','tools.fast_adapter.p67_cup_evaluate','--freeze',str(freeze),'--index',str(index),
            '--output',str(destination),'--device',device]
        log=(output/f'{index:03d}.log').open('w')
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        active[index]=dict(process=process,log=log,started=time.monotonic(),cell=cell,output=str(destination))
    (output/'QUEUE_LAUNCH.json').write_text(json.dumps(dict(freeze_sha256=sha256(freeze),queue=QUEUE,cells=contract['cells'],maximum_attempts=CELL_COUNT),indent=2)+'\n')
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
            records.append(dict(index=index,cell=job['cell'],exit_code=code,output=job['output'],
                elapsed_seconds=time.monotonic()-job['started'],automatic_retries=0))
            if code:failed=True
            elif index==0:first_completed=True
        for index in finished:del active[index]
        if finished:print(json.dumps(dict(completed=len(records),running=len(active),pending=len(pending),program_errors=sum(r['exit_code']!=0 for r in records))),flush=True)
        (output/'QUEUE_PROGRESS.json').write_text(json.dumps(dict(completed=records,running=list(active),pending=[i for i,_ in pending],program_error_seen=failed),indent=2)+'\n')
        if active:time.sleep(.5)
    (output/'QUEUE_COMPLETION.json').write_text(json.dumps(dict(completed=len(records),records=records,unstarted=[i for i,_ in pending],
        maximum_attempts=CELL_COUNT,automatic_retries=0,program_error_seen=failed),indent=2)+'\n')
    if failed:raise RuntimeError('P67programerror: no new submissions; running children awaited; no retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    f=s.add_parser('freeze');f.add_argument('--output',required=True);f.add_argument('--plan',required=True);f.add_argument('--draft',action='store_true')
    q=s.add_parser('queue');q.add_argument('--freeze',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cuda')
    t=s.add_parser('status');t.add_argument('--cells',required=True)
    a=p.parse_args()
    if a.command=='queue':queue(a.freeze,a.output,a.device)
    elif a.command=='status':print(json.dumps(status(a.cells)))
    else:print(json.dumps(prepare(a.output,a.plan,draft=a.draft),indent=2))
