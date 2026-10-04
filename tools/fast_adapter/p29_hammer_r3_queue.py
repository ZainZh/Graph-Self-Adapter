"""Final bounded tool candidate: label correction, matched training, dev only."""
import os,sys,time,json,subprocess
from pathlib import Path
from tools.fast_adapter.p29_common import R,digest,save_json,task_root

def run():
    t=task_root('hammer-block');logs=R/'logs';freeze=t/'candidates/adapter_r3/FREEZE.json'
    for b in json.loads(freeze.read_text())['bindings']:assert digest(b['path'])==b['sha256'],b['path']
    anchor=t/'data/canonical_anchors_r2/COMPLETE.json';assert anchor.exists()
    save_json(logs/'HAMMER_R3_RUNNER_START.json',{'time':time.time(),'pid':os.getpid(),'source':{'path':__file__,'sha256':digest(__file__)},'anchors_sha256':digest(anchor),'before_label_repair':True})
    def start(name,gpu,args):
        env=os.environ.copy();env.update(PYTHONPATH='data/runtime/fast-adapter-p28-deps:.',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',HF_HUB_DISABLE_PROGRESS_BARS='1')
        f=(logs/(name+'.log')).open('x');p=subprocess.Popen([sys.executable,'-m','tools.fast_adapter.p29_hammer_candidate_r3',*args],env=env,stdout=f,stderr=subprocess.STDOUT);print('START',name,gpu,p.pid,flush=True);return p,f
    p,f=start('hammer_r3_labels',2,['tools.fast_adapter.p29_hammer_label_repair']);assert p.wait()==0;f.close()
    jobs=[start('hammer_r3_seed'+str(s),2,['tools.fast_adapter.p29_hammer_r3','train','--seed',str(s)]) for s in (11,22)]
    for p,f in jobs:assert p.wait()==0;f.close()
    files=[freeze,t/'candidates/base_v03/CANDIDATE_V2.json',t/'candidates/base_v03/base.pt',t/'data/PCA.pt',R/'PROTOCOL.json',R/'EVALUATION_SOURCE_FREEZE.json',anchor,t/'data/meta_relabel_r3/COMPLETE.json',Path('tools/fast_adapter/p29_hammer_r3_queue.py')]+[t/'models/r3'/f'seed{s}'/(v+'.pt') for s in (11,22) for v in ('fast','history','robot')]
    bindings=[{'path':str(p),'sha256':digest(p)} for p in files]+json.loads(freeze.read_text())['bindings']+json.loads((R/'EVALUATION_SOURCE_FREEZE.json').read_text())['bindings']+json.loads((t/'candidates/base_v03/CANDIDATE_V2.json').read_text())['bindings']
    fp=t/'DEV_r3_FREEZE.json';assert not fp.exists();save_json(fp,{'time':time.time(),'phase':'development','outcomes_not_opened':True,'round':'r3','bindings':bindings})
    until=time.time()+3600
    while not (logs/'HAMMER_R2_DONE.json').exists():
        assert time.time()<until,'r2 did not finish; inspect instead of racing its dev jobs'
        time.sleep(10)
    jobs=[start('hammer_r3_dev_'+st,gpu,['tools.fast_adapter.p29_eval','--task','hammer-block','--stratum',st,'--round','r3']) for gpu,st in enumerate(('canonical','train','geometry_ood'))]
    for p,f in jobs:assert p.wait()==0;f.close()
    save_json(logs/'HAMMER_R3_DONE.json',{'time':time.time(),'final_not_started':True});print('DONE',flush=True)
if __name__=='__main__':run()
