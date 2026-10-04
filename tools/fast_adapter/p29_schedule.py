"""Finite script-only queue: wait for qualification, train two seeds, run all dev arms.
No outcomes-based model mutation, no API/agent, no final evaluation without new seal.
"""
import os,subprocess,sys,time
from pathlib import Path
from tools.fast_adapter.p29_common import *

def scheduler():
    jobs={};done=set();gpus=(0,1,2);began=time.time();out=R/'logs';out.mkdir(exist_ok=True)
    while time.time()-began<6*3600:
        for name,j in list(jobs.items()):
            if j['p'].poll() is not None:
                code=j['p'].returncode
                save_json(out/f'{name}_END.json',{'name':name,'returncode':code,'time':time.time(),'gpu':j['gpu']});j['f'].close();done.add(name);del jobs[name]
                if code:print('JOB_FAILED',name,code,flush=True)
        occupied={j['gpu'] for j in jobs.values()}
        # Preparation initially occupies GPUs0/1; leave each until its marker or gate stop exists.
        for gpu,task in [(0,TASKS[0]),(1,TASKS[1])]:
            if not (task_root(task)/'READY.json').exists() and not (task_root(task)/'PREPARATION_STOP.json').exists():occupied.add(gpu)
        pending=[]
        for task in TASKS:
            root=task_root(task)
            if not (root/'READY.json').exists():continue
            for seed in SEEDS:
                name=f'train_{task}_{seed}'
                if not (root/'models/r1'/f'seed{seed}'/'COMPLETE.json').exists() and name not in jobs and name not in done:
                    pending.append((name,['tools.fast_adapter.p29_train','adapt','--task',task,'--seed',str(seed)]))
            if all((root/'models/r1'/f'seed{s}'/'COMPLETE.json').exists() for s in SEEDS):
                fp=root/'DEV_r1_FREEZE.json'
                if not fp.exists():
                    files=[root/'models/base_final.pt',root/'data/PCA.pt',R/'PROTOCOL.json',R/'EVALUATION_SOURCE_FREEZE.json']
                    files += [root/'models/r1'/f'seed{s}'/f'{v}.pt' for s in SEEDS for v in ('fast','history','robot')]
                    source=json.loads((R/'EVALUATION_SOURCE_FREEZE.json').read_text())['bindings']
                    save_json(fp,{'time':time.time(),'phase':'development','outcomes_not_opened':True,'bindings':[{'path':str(p),'sha256':digest(p)} for p in files]+source})
                for st in ('canonical','train','geometry_ood'):
                    name=f'dev_{task}_{st}'
                    if not (root/'development/r1'/st/'RESULT.json').exists() and name not in jobs and name not in done:
                        pending.append((name,['tools.fast_adapter.p29_eval','--task',task,'--stratum',st]))
        for gpu in gpus:
            if gpu in occupied or not pending:continue
            name,args=pending.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='data/runtime/fast-adapter-p28-deps:.',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',HF_HUB_DISABLE_PROGRESS_BARS='1')
            f=(out/f'{name}.log').open('x');p=subprocess.Popen([sys.executable,'-m',*args],env=env,stdout=f,stderr=subprocess.STDOUT)
            jobs[name]={'p':p,'gpu':gpu,'f':f};print('LAUNCH',name,gpu,p.pid,flush=True)
        finished=all((task_root(t)/'PREPARATION_STOP.json').exists() or all((task_root(t)/'development/r1'/st/'RESULT.json').exists() for st in ('canonical','train','geometry_ood')) for t in TASKS)
        if finished and not jobs:
            save_json(R/'DEVELOPMENT_DONE.json',{'time':time.time(),'seconds':time.time()-began,'tasks':list(TASKS),'final_not_started':True});print('DEVELOPMENT_DONE',flush=True);return
        time.sleep(10)
    save_json(R/'SCHEDULER_TIMEOUT.json',{'time':time.time(),'active_jobs':list(jobs),'note':'Existing children not killed; inspect before restart.'})
if __name__=='__main__':scheduler()
