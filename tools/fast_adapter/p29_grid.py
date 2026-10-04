"""Seal and execute the predeclared four-GPU final partition for the tool task."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from tools.fast_adapter.p29_common import R,task_root,digest,save_json
TASK='hammer-block'

def freeze():
    root=task_root(TASK);final=root/'FINAL_FREEZE.json';assert final.exists() and not (root/'final').exists()
    pp=R/'BLOCK_EXECUTION_PREPARATION.json';plan=json.loads(pp.read_text())
    for b in plan['bindings']:assert digest(b['path'])==b['sha256'],b['path']
    path=root/'FINAL_GRID_FREEZE.json';assert not path.exists()
    bindings=plan['bindings']+[{'path':str(pp),'sha256':digest(pp)},{'path':'tools/fast_adapter/p29_grid.py','sha256':digest(__file__)}]
    save_json(path,{'time':time.time(),'task':TASK,'before_all_hammer_final_queries':True,'final_freeze_sha256':digest(final),'same_scientific_execution_body':True,'workers':plan['workers'],'bindings':bindings})
    print('GRID_FROZEN',path,flush=True)

def launch(host,dino):
    root=task_root(TASK);fp=root/'FINAL_FREEZE.json';f=json.loads(fp.read_text());gp=root/'FINAL_GRID_FREEZE.json';g=json.loads(gp.read_text())
    assert digest(fp)==g['final_freeze_sha256']
    for b in g['bindings']:assert digest(b['path'])==b['sha256'],b['path']
    starts=R/'logs'/f'GRID_{host}_START.json';assert not starts.exists();jobs=[];receipt=[]
    for w in [x for x in g['workers'] if x['host']==host]:
        command=[('tools.fast_adapter.p29_eval_blocks' if s=='tools.fast_adapter.p29_eval' else s) for s in f['module_command_without_stratum']]
        command+=['--stratum',w['stratum'],'--worker',w['worker'],'--blocks',*map(str,w['blocks']),'--dino',dino]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(w['gpu']),PYTHONPATH='data/runtime/fast-adapter-p28-deps:.',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',HF_HUB_DISABLE_PROGRESS_BARS='1')
        log=(R/'logs'/f'GRID_{w["worker"]}.log').open('x')
        p=subprocess.Popen([sys.executable,'-m',*command],env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log,w['worker']));receipt.append({'worker':w['worker'],'pid':p.pid,'gpu':w['gpu']});print('START',w['worker'],p.pid,flush=True)
    save_json(starts,{'time':time.time(),'grid_sha256':digest(gp),'workers':receipt})
    for p,log,name in jobs:
        code=p.wait();log.close();save_json(R/'logs'/f'GRID_{name}_END.json',{'time':time.time(),'returncode':code});assert code==0,(name,code)
    save_json(R/'logs'/f'GRID_{host}_DONE.json',{'time':time.time(),'workers':receipt});print('HOST_DONE',host,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','launch']);p.add_argument('--host',choices=['local','server']);p.add_argument('--dino',default='assets/dinov3');a=p.parse_args()
    if a.mode=='freeze':freeze()
    else:launch(a.host,a.dino)
