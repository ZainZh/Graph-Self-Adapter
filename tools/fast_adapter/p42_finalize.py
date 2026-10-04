"""Bounded postprocessing after both immutable run queues finish."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.fast_adapter.p41_run import js
from tools.fast_adapter.p42_batch import ROOT


def main():
    deadline=time.time()+7200
    while not all((ROOT/'independent_v01'/f'{task}_DONE.json').exists() for task in ('cup','insert')):
        if time.time()>deadline:
            raise RuntimeError('Bounded two-hour completion wait expired; original workers are not modified')
        time.sleep(10)
    started=time.time()
    js(ROOT/'summary/POSTPROCESS_STATUS.json',dict(stage='all_episode_audits',started=started,pid=os.getpid()))
    workers=[]
    for task in ('cup','insert'):
        log=open(ROOT/'summary'/f'{task}_audit.log','w')
        workers.append((task,subprocess.Popen([sys.executable,'-m','tools.fast_adapter.p42_audit','--task',task],stdout=log,stderr=subprocess.STDOUT),log))
    failures=[]
    for task,worker,log in workers:
        rc=worker.wait();log.close()
        if rc:failures.append(dict(task=task,returncode=rc))
    if failures:
        js(ROOT/'summary/POSTPROCESS_STATUS.json',dict(stage='audit_failed',failures=failures,started=started,ended=time.time()))
        raise RuntimeError(str(failures))
    for stage,args in [('combine',['tools.fast_adapter.p42_audit','--task','combine']),
                       ('statistics',['tools.fast_adapter.p42_stats']),
                       ('media',['tools.fast_adapter.p42_media'])]:
        js(ROOT/'summary/POSTPROCESS_STATUS.json',dict(stage=stage,started=started,pid=os.getpid()))
        subprocess.run([sys.executable,'-m']+args,check=True)
    js(ROOT/'summary/POSTPROCESS_STATUS.json',dict(stage='complete',started=started,ended=time.time(),pid=os.getpid()))


if __name__=='__main__':main()
