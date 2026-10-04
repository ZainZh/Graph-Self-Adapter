"""Finite three-GPU P22 campaign; deterministic gates, no retuning agent."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from tools.fast_adapter.p18_run import save_json,digest
from tools.fast_adapter.p22_data import R

def environment(gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
      MUJOCO_GL='egl',OMP_NUM_THREADS='2',HF_HUB_DISABLE_PROGRESS_BARS='1')
    return env

def run(command,gpu,label):
    log=R/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
    with log.open('x') as f:subprocess.run([sys.executable,'-m',*command],env=environment(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)

def parallel(commands):
    jobs=[]
    for gpu,label,command in commands:
        log=R/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
        stream=log.open('x')
        p=subprocess.Popen([sys.executable,'-m',*command],env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT)
        jobs.append((p,stream,label))
        print(json.dumps({'started':label,'pid':p.pid,'gpu':gpu}),flush=True)
    results=[]
    for p,stream,label in jobs:
        code=p.wait();stream.close();results.append((label,code))
        print(json.dumps({'completed':label,'exit_code':code}),flush=True)
    if any(c for _,c in results):raise RuntimeError(results)

def verify_sources():
    for b in json.loads((R/'SOURCE_FREEZE.json').read_text())['bindings']:
        assert digest(Path(b['path']))==b['sha256'],b['path']

def evaluate_refs(stage):
    parallel([(i,f'{stage}_{st}_reference',['tools.fast_adapter.p22_eval','--stage',stage,'--stratum',st,
               '--seed','0','--dino','assets/dinov3']) for i,st in enumerate(('id','position','scale'))])

def worker(stage,seed,gpu):
    if stage=='dev':run(['tools.fast_adapter.p22_train','adapter','--seed',str(seed)],gpu,f'train_seed{seed}')
    for st in ('id','position','scale'):
        run(['tools.fast_adapter.p22_eval','--stage',stage,'--stratum',st,'--seed',str(seed),'--dino','assets/dinov3'],
            gpu,f'{stage}_{st}_seed{seed}')

def main():
    verify_sources();began=time.time()
    parallel([(g,f'collect_shard{g}',['tools.fast_adapter.p22_data','collect','--shard',str(g),'--dino','assets/dinov3']) for g in range(3)])
    run(['tools.fast_adapter.p22_data','merge'],0,'merge')
    save_json(R/'TRAIN_FREEZE.json',{'time':time.time(),'bindings':[{'path':str(p),'sha256':digest(p)} for p in
      [R/'PROTOCOL.json',R/'SOURCE_FREEZE.json',R/'bank/TRAIN.pt',R/'bank/PCA.pt']]})
    run(['tools.fast_adapter.p22_train','base'],0,'train_base')
    evaluate_refs('dev')
    refs={s:json.loads((R/'dev'/s/'seed0/RESULT.json').read_text())['metrics'] for s in ('id','position','scale')}
    base_ok=refs['id']['base']['successes']>=8 and all(x['oracle']['successes']>=15 for x in refs.values())
    save_json(R/'BASE_GATE.json',{'passed':base_ok,'metrics':refs,'rule':'ID base>=8/16 and oracle>=15/16 each'})
    if not base_ok:
        save_json(R/'CAMPAIGN_DONE.json',{'status':'completed_base_qualification_failed','seconds':time.time()-began,
          'final_run':False,'adapter_trained':False});return
    parallel([(g,f'dev_worker_seed{s}',['tools.fast_adapter.p22_batch','--worker','dev','--seed',str(s),'--gpu',str(g)]) for g,s in enumerate((11,22,33))])
    verify_sources()
    bindings=[R/'PROTOCOL.json',R/'SOURCE_FREEZE.json',R/'bank/PCA.pt',R/'models/base/model.pt']
    for s in (11,22,33):bindings.extend([R/'models'/f'seed{s}'/f'{name}.pt' for name in ('history','hybrid')])
    save_json(R/'FINAL_FREEZE.json',{'time':time.time(),'outcomes_opened':False,'base_qualified':True,
      'no_adapter_result_selection':True,'bindings':[{'path':str(p),'sha256':digest(p)} for p in bindings]})
    evaluate_refs('final')
    parallel([(g,f'final_worker_seed{s}',['tools.fast_adapter.p22_batch','--worker','final','--seed',str(s),'--gpu',str(g)]) for g,s in enumerate((11,22,33))])
    save_json(R/'CAMPAIGN_DONE.json',{'status':'completed_pending_root_audit','seconds':time.time()-began,'final_run':True})
    print('CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['dev','final']);p.add_argument('--seed',type=int);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.worker:worker(a.worker,a.seed,a.gpu)
    else:main()
