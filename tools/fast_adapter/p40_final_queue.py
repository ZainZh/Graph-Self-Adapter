"""Frozen independent P40 queue; three task workers, no online model selection."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from tools.fast_adapter.p40_launch_v03 import ROOT,environment
TASKS=('cup','hammer','insert')

def worker(task):
    freeze=json.loads((ROOT/'FINAL_FREEZE.json').read_text());cells=[c for c in freeze['cells'] if c['task']==task]
    for i,c in enumerate(cells):
        tag='final_'+task+'_'+c['split']+'_'+c['kind']+'_'+str(c['seed'])+'_'+c['arm']
        args=[sys.executable,'-u','-m','tools.fast_adapter.p40_evaluate_v02','--phase','final']
        for k,v in c.items():args+=['--'+k,str(v)]
        with (ROOT/'logs'/(tag+'.log')).open('x') as f:subprocess.run(args,env=environment(TASKS.index(task)),stdout=f,stderr=subprocess.STDOUT,check=True)
        (ROOT/task/'FINAL_PROGRESS.json').write_text(json.dumps(dict(time=time.time(),completed=i+1,total=len(cells))))
    (ROOT/task/'FINAL_DONE.json').write_text(json.dumps(dict(time=time.time(),cells=len(cells))))

def main():
    freeze=json.loads((ROOT/'FINAL_FREEZE.json').read_text())
    for b in freeze['sources']+freeze['models']+freeze['visual_weights']:assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256'],b['path']
    with (ROOT/'FINAL_START.json').open('x') as f:json.dump(dict(time=time.time(),pid=os.getpid(),bindings_verified=True),f)
    jobs=[]
    for task in TASKS:
        with (ROOT/'logs'/f'FINAL_WORKER_{task}.log').open('x') as f:jobs.append(subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p40_final_queue','--task',task],stdout=f,stderr=subprocess.STDOUT))
    exits=[p.wait() for p in jobs];(ROOT/'FINAL_DONE.json').write_text(json.dumps(dict(time=time.time(),exits=exits)))
    if any(exits):raise RuntimeError(exits)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task');a=p.parse_args();worker(a.task) if a.task else main()
