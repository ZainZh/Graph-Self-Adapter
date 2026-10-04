"""Bounded asynchronous development queue, with pre-execution model bindings."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from tools.fast_adapter.p40_launch_v03 import ROOT,environment
from tools.fast_adapter.p40_evaluate import STRATA,configurations

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();records=[]
    for kind in ('relation','local','fused','history'):
        for seed in (11,22):
            model=ROOT/a.task/'models_v03'/kind/f'seed{seed}'/'model.pt';done=model.parent/'TRAIN.json';deadline=time.time()+7200
            while not done.exists():
                if time.time()>deadline:raise RuntimeError(f'Model timeout: {done}')
                time.sleep(5)
            for split in STRATA:
                out=ROOT/a.task/'dev'/kind/f'seed{seed}'/split;out.mkdir(parents=True,exist_ok=True)
                sources=[model,ROOT/a.task/'effects_v03/local.pt',Path('tools/fast_adapter/p40_evaluate.py'),Path('prove/fast_adapter/skill_tokens_v02.py'),Path('prove/fast_adapter/contact_skills.py'),Path('prove/fast_adapter/skill_observation.py')]
                binding=dict(time=time.time(),configs=configurations(a.task,'dev',split),sources=[dict(path=str(p),sha256=digest(p)) for p in sources])
                with (out/'continuous_PRE_RUN.json').open('x') as f:json.dump(binding,f,indent=2)
                tag=f'dev_{a.task}_{kind}_{seed}_{split}_continuous'
                with (ROOT/'logs'/(tag+'.log')).open('x') as f:
                    subprocess.run([sys.executable,'-u','-m','tools.fast_adapter.p40_evaluate','--task',a.task,'--phase','dev','--split',split,'--kind',kind,'--seed',str(seed),'--arm','continuous'],env=environment(a.gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
                records.append(str(out/'continuous_RESULT.json'))
    (ROOT/a.task/'DEV_MODELS_DONE.json').write_text(json.dumps(dict(time=time.time(),records=records),indent=2))

if __name__=='__main__':main()
