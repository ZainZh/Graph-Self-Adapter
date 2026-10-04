"""Exact-root-reviewed recovery of unstarted P67 cells; never retry physics."""
import os
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',EGL_DEVICE_ID='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from prove.fast_adapter.p61_data import sha256
from tools.fast_adapter.p67_cup_evaluate import validate


def destination(root,index,cell):
    return root/f"{index:03d}_{cell['split']}_{cell['seed']}_{cell['arm']}"


def verify_completed(root,freeze,contract,indices):
    """Hash sealed bytes; never deserialize task outcomes or trajectories."""
    receipts=[]
    required={'TRACE.npy','OBS.npy','TERMINAL.npy','PRE_RUN.json','PUBLIC_IMAGES.npz','LATENT_LIFECYCLE.npy'}
    for index in indices:
        cell=contract['cells'][index];directory=destination(root,index,cell)
        if not all((directory/n).is_file() for n in ('PRE_RUN.json','SEAL.json','RESULT.json')):
            raise ValueError(f'Incomplete completed cell {index}')
        if (directory/'PROGRAM_ERROR.json').exists():raise ValueError(f'Program error cell {index}')
        pre=json.loads((directory/'PRE_RUN.json').read_text())
        if any(pre.get(k)!=v for k,v in dict(index=index,cell=cell,freeze_sha256=sha256(freeze),runtime_truth=False,steps=240).items()):
            raise ValueError(f'Completed identity mismatch {index}')
        seal=json.loads((directory/'SEAL.json').read_text())
        if set(seal['files'])!=required:raise ValueError(f'Unexpected seal files {index}')
        for name,digest in seal['files'].items():
            if sha256(directory/name)!=digest:raise ValueError(f'Sealed bytes changed {index}/{name}')
        receipts.append(dict(index=index,pre_run_sha256=sha256(directory/'PRE_RUN.json'),seal_sha256=sha256(directory/'SEAL.json')))
    return receipts


def qualify(path):
    path=Path(path);amendment=json.loads(path.read_text())
    review=json.loads((path.parent/'ROOT_RECOVERY_REVIEW_V01.json').read_text())
    if review.get('approved') is not True or review.get('reviewer')!='root' or review.get('amendment_sha256')!=sha256(path):
        raise ValueError('Exact root recovery review required')
    if amendment.get('schema')!='prove.p67-recovery-amendment.v1':raise ValueError('Wrong recovery schema')
    if amendment.get('completed_indices')!=list(range(47)) or amendment.get('pending_indices')!=list(range(47,768)):
        raise ValueError('Recovery scope differs from declared unstarted cells')
    freeze=Path(amendment['freeze'])
    if sha256(freeze)!=amendment['freeze_sha256']:raise ValueError('Original freeze changed')
    bindings=amendment['files'];resolved={str(Path(p).resolve()) for p in bindings}
    for required in (__file__,Path(__file__).resolve().parents[2]/'tests/test_p67_resume.py'):
        if str(Path(required).resolve()) not in resolved:raise ValueError('Recovery source binding missing')
    for name,digest in bindings.items():
        if sha256(name)!=digest:raise ValueError('Recovery bound source changed: '+name)
    contract=validate(freeze);root=Path(amendment['cells_root'])
    receipts=verify_completed(root,freeze,contract,amendment['completed_indices'])
    expected={destination(root,i,c).name for i,c in enumerate(contract['cells'][:47])}
    if {p.name for p in root.iterdir() if p.is_dir()}!=expected:raise ValueError('Unexpected/partial/concurrent cell directory')
    for index in amendment['pending_indices']:
        if destination(root,index,contract['cells'][index]).exists():raise ValueError('Pending cell already exists')
    for key in ('resume_root','report_output','presentation_output'):
        if Path(amendment[key]).exists():raise FileExistsError(amendment[key])
    return amendment,contract,receipts


def run(path,device='cuda'):
    amendment,contract,receipts=qualify(path);root=Path(amendment['cells_root']);out=Path(amendment['resume_root'])
    out.mkdir(parents=True,exist_ok=False);logs=out/'logs';logs.mkdir()
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    write('RECOVERY_LAUNCH.json',dict(amendment_sha256=sha256(path),completed_receipts=receipts,pending_indices=amendment['pending_indices'],automatic_retries=0))
    pending=list(amendment['pending_indices']);active={};records=[];failed=False;first_done=False
    while pending or active:
        cap=2 if first_done else 1
        while pending and not failed and len(active)<cap:
            index=pending.pop(0);cell=contract['cells'][index];dest=destination(root,index,cell)
            # Repeat immediately before submission: a competing recovery is an error.
            if dest.exists():
                failed=True;records.append(dict(index=index,exit_code=-1,error='pending destination exists'));break
            log=(logs/f'{index:03d}.log').open('x')
            command=[sys.executable,'-m','tools.fast_adapter.p67_cup_evaluate','--freeze',amendment['freeze'],'--index',str(index),'--output',str(dest),'--device',device]
            try:process=subprocess.Popen(command,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
            except Exception as error:
                log.close();failed=True;records.append(dict(index=index,exit_code=-1,error=str(error)));break
            active[index]=(process,log)
            if not first_done:break
        if not active:break
        for index,(process,log) in list(active.items()):
            code=process.poll()
            if code is None:continue
            log.close();del active[index];records.append(dict(index=index,exit_code=code))
            if code:failed=True
            elif index==47:first_done=True
        write('RECOVERY_PROGRESS.json',dict(completed=records,running=list(active),pending=pending,program_error_seen=failed))
        if active:time.sleep(.5)
    write('RECOVERY_COMPLETION.json',dict(completed=records,unstarted=pending,program_error_seen=failed,automatic_retries=0))
    if failed:raise RuntimeError('Recovery program error; submissions stopped, running children awaited; no retry')
    verify_completed(root,Path(amendment['freeze']),contract,range(768))
    # Reporting failures never trigger physics. Both tools operate only after all seals.
    commands=[['tools.fast_adapter.p67_report','--cells',str(root),'--freeze',amendment['freeze'],'--output',amendment['report_output']],
              ['tools.fast_adapter.p67_presentation','--report',str(Path(amendment['report_output'])/'RESULTS.json'),'--output',amendment['presentation_output']]]
    for ordinal,arguments in enumerate(commands):
        with (out/f'postprocess_{ordinal}.log').open('x') as log:
            result=subprocess.run([sys.executable,'-m',*arguments],env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write('POSTPROCESS_ERROR.json',dict(command=arguments,exit_code=result.returncode,physics_retry=False))
            raise RuntimeError('Postprocessing failed; physical records retained; do not rerun physics')
    write('POSTPROCESS_COMPLETION.json',dict(report=amendment['report_output'],presentation=amendment['presentation_output']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--amendment',required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run(args.amendment,args.device)
