"""P39 parallel evaluation only, requires a separately frozen phase."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.fast_adapter.p39_launch import ROOT,env

def run(kind,seed,phase,gpu,split=None):
    args=['--kind',kind,'--seed',str(seed),'--phase',phase]
    if split:args+=['--split',split]
    tag=f'{phase}_{kind}_{seed}'+('_'+split if split else '')
    with (ROOT/'logs'/(tag+'.log')).open('x') as f:
        subprocess.run([sys.executable,'-u','-m','tools.fast_adapter.p39_evaluate']+args,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)

def worker(gpu,phase):
    kinds=[('global','history'),('local',),('relation',),('fused',)][gpu]
    for kind in kinds:
        for seed in (11,22):
            deadline=time.time()+1800
            checkpoint=ROOT/'models'/kind/f'seed{seed}'/'TRAIN.json'
            while not checkpoint.exists():
                if time.time()>deadline:raise TimeoutError(str(checkpoint))
                done=ROOT/'TRAIN_DONE.json'
                if done.exists() and any(json.loads(done.read_text())['exits']):raise RuntimeError('Training failed')
                time.sleep(15)
            run(kind,seed,phase,gpu)

def launch(phase):
    assert (ROOT/(phase.upper()+'_FREEZE.json')).exists()
    if phase=='final':assert (ROOT/'TRAIN_DONE.json').exists()
    with (ROOT/(phase.upper()+'_START.json')).open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
    # Reference passes first produce fixed donors and common baseline controls.
    run('references',0,phase,0)
    if not (ROOT/'TRAIN_DONE.json').exists():
        raise RuntimeError('Training incomplete. References retained; model launch must be resumed explicitly.')
    models(phase)

def models(phase):
    jobs=[]
    for gpu in range(4):
        with (ROOT/'logs'/f'{phase}_worker{gpu}.log').open('x') as f:
            jobs.append(subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p39_eval_launch','--phase',phase,'--worker',str(gpu)],stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    (ROOT/(phase.upper()+'_DONE.json')).write_text(json.dumps(dict(time=time.time(),exits=codes)))
    if any(codes):raise RuntimeError(codes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','final'],default='dev');p.add_argument('--worker',type=int);p.add_argument('--models-only',action='store_true');a=p.parse_args()
    if a.worker is not None:worker(a.worker,a.phase)
    elif a.models_only:models(a.phase)
    else:launch(a.phase)
