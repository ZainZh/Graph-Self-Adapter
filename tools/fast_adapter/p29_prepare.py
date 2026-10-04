import argparse,os,subprocess,sys,time
from pathlib import Path
from tools.fast_adapter.p29_common import *

def run_module(module,*args):
    subprocess.run([sys.executable,'-m',module,*map(str,args)],check=True)
def qualify_base(task,dino):
    from tools.fast_adapter.p29_data import collect_block
    out=task_root(task)/'qualification';out.mkdir(parents=True,exist_ok=True)
    base=load_base(task);v=Visual(dino,task,torch.load(task_root(task)/'data/PCA.pt',weights_only=True))
    cfgs=[object_config(task,2940000+i,'canonical') for i in range(8)];seeds=[2941000+TASKS.index(task)*1000+i for i in range(8)]
    blob,metrics=collect_block(task,cfgs,seeds,HORIZON,'base',v,base,labels=False)
    path=out/'BASE_RAW.pt';torch.save(blob,path)
    rows=[{'reset_seed':s,'success_any':bool(any(x['success'] for x in m)),'terminal_success':bool(m[-1]['success'])} for s,m in zip(seeds,metrics)]
    count=sum(x['success_any'] for x in rows);report={'task':task,'successes':count,'n':8,'passed':count>=6,'rows':rows,'source':str(path),'sha256':digest(path)}
    save_json(out/'BASE.json',report);save_json(out/'BASE_EVALUATOR.json',{'configs':cfgs,'metrics':metrics,'source_sha256':report['sha256']});print(task,'BASE_GATE',count,'/8',flush=True)
    return report['passed']
def prepare(task,dino):
    root=task_root(task);began=time.time();root.mkdir(parents=True,exist_ok=True)
    commands=[('data/demos/COMPLETE.json','tools.fast_adapter.p29_data','collect','--task',task,'--stage','demos','--dino',dino),
      ('data/PCA_SEAL.json','tools.fast_adapter.p29_data','merge','--task',task),
      ('models/BASE_bc_COMPLETE.json','tools.fast_adapter.p29_train','base','--task',task,'--stage','bc'),
      ('data/dagger/COMPLETE.json','tools.fast_adapter.p29_data','collect','--task',task,'--stage','dagger','--dino',dino),
      ('models/BASE_dagger_COMPLETE.json','tools.fast_adapter.p29_train','base','--task',task,'--stage','dagger')]
    for marker,module,*args in commands:
        if not (root/marker).exists():run_module(module,*args)
    if not (root/'qualification/BASE.json').exists():passed=qualify_base(task,dino)
    else:passed=json.loads((root/'qualification/BASE.json').read_text())['passed']
    if not passed:
        save_json(root/'PREPARATION_STOP.json',{'reason':'base_not_qualified','seconds':time.time()-began});return 2
    if not (root/'data/meta/COMPLETE.json').exists():run_module('tools.fast_adapter.p29_data','collect','--task',task,'--stage','meta','--dino',dino)
    save_json(root/'READY.json',{'task':task,'seconds':time.time()-began,'time':time.time(),'base_qualified':True})
    print(task,'PREPARATION_READY',flush=True);return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--dino',default='assets/dinov3');a=p.parse_args();raise SystemExit(prepare(a.task,a.dino))
