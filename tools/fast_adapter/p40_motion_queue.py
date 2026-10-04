"""Finite extra representation candidate, six models and development only."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from tools.fast_adapter.p40_launch_v03 import ROOT,environment
from tools.fast_adapter.p40_evaluate_v02 import STRATA,configurations
TASKS=('cup','hammer','insert')

def run(module,args,gpu,tag):
    with (ROOT/'logs'/(tag+'.log')).open('x') as f:subprocess.run([sys.executable,'-u','-m',module]+args,env=environment(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)

def worker(task):
    gpu={'cup':3,'hammer':0,'insert':1}[task]
    dependency={'cup':('history',22),'hammer':('relation',22),'insert':('local',22)}[task]
    deadline=time.time()+7200
    while not (ROOT/'insert/models_v03'/dependency[0]/f'seed{dependency[1]}'/'TRAIN.json').exists():
        if time.time()>deadline:raise RuntimeError('Original model queue timeout')
        time.sleep(5)
    for seed in (11,22):run('tools.fast_adapter.p40_motion',['train','--task',task,'--seed',str(seed)],gpu,f'motion_train_{task}_{seed}')
    run('tools.fast_adapter.p40_motion',['mean','--task',task],gpu,f'motion_mean_{task}')
    for seed in (11,22):
        for split in STRATA:
            out=ROOT/task/'dev/motion'/f'seed{seed}'/split;out.mkdir(parents=True,exist_ok=True)
            paths=[ROOT/task/'motion'/f'seed{seed}'/'model.pt',ROOT/task/'motion/local.pt',ROOT/task/'motion/PCA.pt',Path('tools/fast_adapter/p40_evaluate_v02.py'),Path('prove/fast_adapter/skill_visual_reference.py')]
            with (out/'continuous_PRE_RUN.json').open('x') as f:json.dump(dict(time=time.time(),configs=configurations(task,'dev',split),sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]),f,indent=2)
            run('tools.fast_adapter.p40_evaluate_v02',['--task',task,'--phase','dev','--split',split,'--kind','motion','--seed',str(seed),'--arm','continuous'],TASKS.index(task),f'motion_dev_{task}_{seed}_{split}')
    (ROOT/task/'MOTION_DEV_DONE.json').write_text(json.dumps(dict(time=time.time())))

def main():
    (ROOT/'MOTION_START.json').write_text(json.dumps(dict(time=time.time())))
    for task in TASKS:run('tools.fast_adapter.p40_motion',['prepare','--task',task],3,f'motion_prepare_{task}')
    jobs=[]
    for task in TASKS:
        with (ROOT/'logs'/f'MOTION_WORKER_{task}.log').open('x') as f:jobs.append(subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p40_motion_queue','--task',task],stdout=f,stderr=subprocess.STDOUT))
    exits=[p.wait() for p in jobs];(ROOT/'MOTION_DONE.json').write_text(json.dumps(dict(time=time.time(),exits=exits)))
    if any(exits):raise RuntimeError(exits)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task');a=p.parse_args();worker(a.task) if a.task else main()
