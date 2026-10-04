"""Finite GPU queues; does not tune, add trials, or invoke any learning agent."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p21_frontends_v01')

def worker(stage,seed,gpu):
    selection=json.loads((R/'diagnostic/RESULT.json').read_text())
    if not selection['control_qualified']:return
    reps=selection['control_frontends'];env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='2',HF_HUB_DISABLE_PROGRESS_BARS='1')
    began=time.time()
    for rep in reps:
        commands=[]
        if stage=='dev':commands.append([sys.executable,'-m','tools.fast_adapter.p21_control','train','--frontend',rep,'--seed',str(seed)])
        for st in (['train'] if stage=='dev' else ['train','ood']):
            commands.append([sys.executable,'-m','tools.fast_adapter.p21_control','eval','--frontend',rep,'--seed',str(seed),'--stage',stage,'--stratum',st,'--dino','assets/dinov3'])
        for command in commands:subprocess.run(command,env=env,check=True)
    save_json(R/f'{stage.upper()}_WORKER_seed{seed}_DONE.json',{'seed':seed,'gpu':gpu,'stage':stage,'seconds':time.time()-began,'frontends':reps,'completed':True})

def batch(stage):
    selection=json.loads((R/'diagnostic/RESULT.json').read_text())
    if not selection['control_qualified']:
        print('STOP_FEATURE_QUALIFICATION_NEGATIVE',flush=True);return
    if stage=='final':assert (R/'FINAL_FREEZE.json').exists()
    if stage=='dev':
        frozen=[R/'PROTOCOL.json',R/'diagnostic/RESULT.json']
        for rep in selection['control_frontends']:
            frozen += [R/'frontends'/rep/'bank'/name for name in ['TRAIN.pt','PCA.pt']]
        frozen += [Path('tools/fast_adapter')/name for name in ['p21_control.py','p21_batch.py','p21_results.py','p20_train.py','p20_eval.py']]
        save_json(R/'TRAIN_FREEZE.json',{'created_at_unix':time.time(),'scope':'before any P21 training/control outcomes','selected':selection['selected'],'bindings':[{'path':str(p),'sha256':digest(p)} for p in frozen]})
    jobs=[]
    for gpu,seed in enumerate([11,22,33]):
        logfile=R/f'{stage}_seed{seed}.log'
        p=subprocess.Popen([sys.executable,'-m','tools.fast_adapter.p21_batch','--stage',stage,'--worker',str(seed),'--gpu',str(gpu)],stdout=open(logfile,'w'),stderr=subprocess.STDOUT)
        jobs.append((p,{'seed':seed,'gpu':gpu,'pid':p.pid,'log':str(logfile)}))
    save_json(R/f'{stage.upper()}_JOBS.json',{'jobs':[r for _,r in jobs],'stage':stage})
    for st in (['train'] if stage=='dev' else ['train','ood']):
        with open(R/f'{stage}_references_{st}.log','w') as log:
            subprocess.run([sys.executable,'-m','tools.fast_adapter.p21_control','eval','--frontend','last','--seed','0','--stage',stage,'--stratum',st],stdout=log,stderr=subprocess.STDOUT,check=True)
    codes=[p.wait() for p,_ in jobs]
    save_json(R/f'{stage.upper()}_BATCH_DONE.json',{'stage':stage,'codes':codes,'completed':all(c==0 for c in codes)})
    if not all(c==0 for c in codes):raise RuntimeError(codes)
    subprocess.run([sys.executable,'-m','tools.fast_adapter.p21_results',stage],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='dev',choices=['dev','final']);p.add_argument('--worker',type=int);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.worker:worker(a.stage,a.worker,a.gpu)
    else:batch(a.stage)
