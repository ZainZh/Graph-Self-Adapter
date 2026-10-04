"""Wait for finite task queues, then prepare controls and audit development."""
import argparse,json,subprocess,sys,time
from tools.fast_adapter.p40_launch_v03 import ROOT,environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();deadline=time.time()+3600
    while not all((ROOT/a.task/p).exists() for p in ('DEV_MODELS_DONE.json','MOTION_DEV_DONE.json')):
        if time.time()>deadline:raise RuntimeError('Development dependency timeout')
        time.sleep(5)
    for module,args,tag in (
        ('tools.fast_adapter.p40_evaluate_v02',['--task',a.task,'--means'],'means'),
        ('tools.fast_adapter.p40_audit_v02',['--task',a.task,'--phase','dev'],'audit')):
        if tag=='means' and (ROOT/a.task/'MEANS_v03.pt').exists():continue
        # Preserve the earlier bounded cup audit when the motion candidate was incomplete.
        current=ROOT/a.task/'dev_AUDIT.json';earlier=ROOT/a.task/'dev_AUDIT_partial_original.json'
        if tag=='audit' and current.exists() and not earlier.exists():current.rename(earlier)
        with (ROOT/'logs'/f'dev_close_{a.task}_{tag}.log').open('x') as f:
            subprocess.run([sys.executable,'-u','-m',module]+args,env=environment(a.gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
    (ROOT/a.task/'DEV_ACCEPTABLE.json').write_text(json.dumps(dict(time=time.time(),numerical_audit_passed=True,scope='Integrity accepted; scientific outcomes require separate review')))

if __name__=='__main__':main()
