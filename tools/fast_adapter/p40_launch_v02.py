"""Finite P40 training queue; three collection workers, four training workers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('evaluations/fast_adapter_p40_skills_v01')
TASKS=('cup','hammer','insert');KINDS=('relation','local','fused','history')

def environment(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1');return e

def run(args,gpu,tag):
    with (ROOT/'logs'/('v02_'+tag+'.log')).open('x') as f:
        subprocess.run([sys.executable,'-u','-m','tools.fast_adapter.p40_train_v02']+args,env=environment(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)

def collect(gpu):
    task=TASKS[gpu]
    for mode in ('collect','prepare','effects'):run([mode,'--task',task],gpu,mode+'_'+task)

def worker(gpu):
    for task in TASKS:
        for seed in (11,22):run(['train','--task',task,'--kind',KINDS[gpu],'--seed',str(seed)],gpu,f'train_{task}_{KINDS[gpu]}_{seed}')

def main():
    js=lambda name,x:(ROOT/name).write_text(json.dumps(x,indent=2))
    with (ROOT/'TRAIN_v02_START.json').open('x') as f:json.dump(dict(time=time.time(),pid=os.getpid()),f)
    for stage,indices in (('collect',[2]),('worker',range(4))):
        if stage=='worker':
            deadline=time.time()+1800
            while not all((ROOT/task/'effects/DONE.json').exists() for task in TASKS):
                if time.time()>deadline:raise RuntimeError('Cup/hammer data dependency timeout; do not train partial banks')
                time.sleep(10)
        jobs=[]
        for gpu in indices:
            with (ROOT/'logs'/f'v02_{stage}_worker{gpu}.log').open('x') as f:
                jobs.append(subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p40_launch_v02','--'+stage,str(gpu)],stdout=f,stderr=subprocess.STDOUT))
        exits=[p.wait() for p in jobs];js(stage.upper()+'_v02_DONE.json',dict(time=time.time(),exits=exits))
        if any(exits):raise RuntimeError((stage,exits))
    js('TRAIN_v02_DONE.json',dict(time=time.time(),status='complete'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect',type=int);p.add_argument('--worker',type=int);a=p.parse_args()
    if a.collect is not None:collect(a.collect)
    elif a.worker is not None:worker(a.worker)
    else:main()
