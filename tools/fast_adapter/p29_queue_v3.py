"""Continue P29 after original base gates; finite jobs, no test-result-driven edits."""
import os,sys,time,subprocess
from pathlib import Path
from tools.fast_adapter.p29_common import *

def run():
    jobs={};ended=set();out=R/'logs';began=time.time();mug,hammer=TASKS
    while time.time()-began<6*3600:
        for name,j in list(jobs.items()):
            if j['p'].poll() is not None:
                code=j['p'].returncode;j['f'].close();ended.add(name);del jobs[name]
                save_json(out/f'{name}_END.json',{'time':time.time(),'returncode':code,'gpu':j['gpu']});print('END',name,code,flush=True)
        busy={j['gpu'] for j in jobs.values()}
        external_markers=set()
        for gpu,st,pid in [(1,'canonical',1898333),(2,'train',1898334)]:
            marker=task_root(mug)/'development/r1'/st/'RESULT.json'
            if not marker.exists():
                try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                except FileNotFoundError:cmd=b''
                if b'tools.fast_adapter.p29_eval' in cmd:
                    busy.add(gpu);external_markers.add(marker)
                else:raise RuntimeError(f'Adopted development worker {pid} ended without its result; inspect, do not restart into existing artifacts.')
        for gpu,seed in [(0,11),(1,22)]:
            if not (task_root(mug)/'models/r1'/f'seed{seed}'/'COMPLETE.json').exists():busy.add(gpu)
        gate_path=task_root(hammer)/'candidates/base_v03/BASE_GATE.json'
        if not gate_path.exists():busy.add(2)
        gate=json.loads(gate_path.read_text()) if gate_path.exists() else None
        candidate_path=task_root(hammer)/'candidates/base_v03/CANDIDATE_V2.json'
        if gate and gate['passed'] and not candidate_path.exists():
            files=[task_root(hammer)/'candidates/base_v03/base.pt',task_root(hammer)/'data/PCA.pt',Path('prove/fast_adapter/object_sequence_skill.py'),Path('tools/fast_adapter/p29_candidate_v04.py'),Path('tools/fast_adapter/p29_recurrent_train.py')]
            save_json(candidate_path,{'time':time.time(),'task':hammer,'base_kind':'recurrent','base_checkpoint':str(files[0]),'meta_stage':'meta','base_gate':str(gate_path),'ordinary_causal_query_state':True,'bindings':[{'path':str(p),'sha256':digest(p)} for p in files]})
        pending=[]
        def add(name,args,marker):
            if not marker.exists() and marker not in external_markers and name not in jobs and name not in ended:pending.append((name,args))
        prefix=['tools.fast_adapter.p29_candidate_v04','--candidate',str(candidate_path)]
        if gate and gate['passed']:
            if not (task_root(hammer)/'data/meta/COMPLETE.json').exists():
                add('v3_hammer_meta',prefix+['tools.fast_adapter.p29_data','collect','--task',hammer,'--stage','meta'],task_root(hammer)/'data/meta/COMPLETE.json')
            else:
                for seed in SEEDS:add(f'v3_train_hammer_{seed}',prefix+['tools.fast_adapter.p29_recurrent_train','adapt','--task',hammer,'--seed',str(seed)],task_root(hammer)/'models/r1'/f'seed{seed}'/'COMPLETE.json')
        eligible=[mug]+([hammer] if gate and gate['passed'] else [])
        for task in eligible:
            root=task_root(task)
            if all((root/'models/r1'/f'seed{s}'/'COMPLETE.json').exists() for s in SEEDS):
                fp=root/'DEV_r1_FREEZE.json'
                if not fp.exists():
                    files=[root/'models/base_final.pt',root/'data/PCA.pt',R/'PROTOCOL.json',R/'EVALUATION_SOURCE_FREEZE.json',Path('tools/fast_adapter/p29_queue_v3.py')]
                    files += [root/'models/r1'/f'seed{s}'/f'{v}.pt' for s in SEEDS for v in ('fast','history','robot')]
                    bindings=json.loads((R/'EVALUATION_SOURCE_FREEZE.json').read_text())['bindings']
                    if task==hammer:bindings+=json.loads(candidate_path.read_text())['bindings'];files += [candidate_path,gate_path]
                    save_json(fp,{'time':time.time(),'phase':'development','outcomes_not_opened':True,'base_candidate':str(candidate_path) if task==hammer else None,'bindings':[{'path':str(p),'sha256':digest(p)} for p in files]+bindings})
                for st in ('canonical','train','geometry_ood'):
                    args=['tools.fast_adapter.p29_eval','--task',task,'--stratum',st]
                    if task==hammer:args=prefix+args
                    add(f'v3_dev_{task}_{st}',args,root/'development/r1'/st/'RESULT.json')
        for gpu in (0,1,2):
            if gpu in busy or not pending:continue
            name,args=pending.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='data/runtime/fast-adapter-p28-deps:.',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',HF_HUB_DISABLE_PROGRESS_BARS='1')
            f=(out/f'{name}.log').open('x');p=subprocess.Popen([sys.executable,'-m',*args],env=env,stdout=f,stderr=subprocess.STDOUT);jobs[name]={'p':p,'f':f,'gpu':gpu};print('LAUNCH',name,gpu,p.pid,flush=True)
        finished=bool(gate) and all(all((task_root(t)/'development/r1'/st/'RESULT.json').exists() for st in ('canonical','train','geometry_ood')) for t in eligible)
        if finished and not jobs:
            save_json(R/'DEVELOPMENT_DONE.json',{'time':time.time(),'seconds':time.time()-began,'qualified_tasks':eligible,'hammer_base_gate':str(gate_path),'final_not_started':True});print('DEVELOPMENT_DONE',flush=True);return
        time.sleep(10)
    save_json(R/'QUEUE_V3_TIMEOUT.json',{'time':time.time(),'active':list(jobs)})
if __name__=='__main__':run()
