#!/usr/bin/env python3
"""One finite post-pipeline launcher; no retries or success-based arm selection."""
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tools.task_graph.p53_pipeline import execute
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p52_pack_dataset import sha

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError,ProcessLookupError):return None


def prepare(pipeline_freeze,pipeline_sha,pipeline_output,pipeline_pid,parent):
    pipeline_freeze=Path(pipeline_freeze).resolve();parent=Path(parent).resolve();pipeline_output=Path(pipeline_output).resolve()
    if sha(pipeline_freeze)!=pipeline_sha:raise ValueError('Explicit parent pipeline hash mismatch')
    pipeline=read(pipeline_freeze)
    if pipeline['schema']!='p53.finite-pipeline.v1':raise ValueError('Wrong parent pipeline')
    return dict(schema='p53.finite-full-successor.v1',pipeline_freeze=str(pipeline_freeze),pipeline_sha256=pipeline_sha,
        pipeline_output=str(pipeline_output),pipeline_pid=pipeline_pid,pipeline_process_start=process_identity(pipeline_pid),
        parent=str(parent),parent_sha256=sha(parent),config=pipeline['config'],
        sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/p53_full_successor.py','tools/task_graph/p53_full_task.py','tools/task_graph/p52_vision_worker.py'])},
        deadline_seconds=14400,poll_seconds=15,episode_timeout_seconds=1800,cells=['bc','fixed','online'],gpus=[0,1,2],automatic_retries=0)


def verify(plan):
    if plan['schema']!='p53.finite-full-successor.v1' or plan['deadline_seconds']!=14400 or plan['poll_seconds']!=15 or plan['episode_timeout_seconds']!=1800 or plan['cells']!=['bc','fixed','online'] or plan['gpus']!=[0,1,2] or plan['automatic_retries']!=0:raise ValueError('Changed bounded successor scope')
    if sha(plan['pipeline_freeze'])!=plan['pipeline_sha256'] or sha(plan['parent'])!=plan['parent_sha256']:raise ValueError('Frozen dependency changed')
    if plan['config']!=read(plan['pipeline_freeze'])['config']:raise ValueError('Parent runtime configuration changed')
    for p,h in plan['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen successor source changed:'+p)


def closed_gate(output):
    summary=read(output/'COMPLETE.json')
    if any(summary.get(k)!=v for k,v in dict(expected_cells=36,closed_queue_attempts=36,sealed_scientific_results=36,engineering_aborts=0).items()):raise ValueError('Parent not36sealed/zeroengineeringabort')
    queue=read(output/'eval/QUEUE_RESULT.json');seen=set()
    if len(queue['attempts'])!=36:raise ValueError('Incomplete closed queue')
    for receipt in queue['attempts']:
        if receipt.get('returncode')!=0 or receipt.get('execution_status')!='finished':raise ValueError('Parent queue engineering failure')
        folder=output/'eval'/receipt['job']['id']/'episode';r=read(folder/'RESULT.json');seal=read(folder/'SEAL.json')
        if sha(folder/'SEAL.json')!=r['seal_sha256'] or sha(folder/'BINDING.json')!=seal['binding_sha256']:raise ValueError('Parent result/seal mismatch')
        seen.add(r['cell']['index'])
        for name,h in seal['files'].items():
            if sha(folder/name)!=h:raise ValueError('Parent sealed evidence changed')
    if seen!=set(range(36)):raise ValueError('Parent cell identities incomplete')
    # Never inspect successes to select arms. Failed scientific outcomes pass.
    return dict(complete_sha256=sha(output/'COMPLETE.json'),queue_sha256=sha(output/'eval/QUEUE_RESULT.json'),cells=36,engineering_aborts=0)


def run(plan_path,output):
    plan=read(plan_path);verify(plan);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/'PLAN.json',plan);deadline=time.monotonic()+plan['deadline_seconds'];parent=Path(plan['pipeline_output']);c=plan['config']
    def blocked(reason):write(output/'BLOCKED.json',dict(reason=reason,scientific_failure=False,automatic_retry=False))
    try:
        while True:
            verify(plan)
            if (parent/'STOP.json').exists() or any(parent.rglob('*ABORT.json')):blocked('Parent pipeline STOP/ABORT');return
            if (parent/'COMPLETE.json').exists():break
            if time.monotonic()>=deadline:blocked('Four-hour dependency deadline');return
            identity=process_identity(plan['pipeline_pid'])
            if identity is None or identity!=plan['pipeline_process_start']:blocked('Parent process exited/replaced before COMPLETE');return
            time.sleep(min(plan['poll_seconds'],max(0,deadline-time.monotonic())))
        try:gate=closed_gate(parent)
        except (ValueError,OSError,KeyError) as e:blocked('Parent closure invalid: '+str(e));return
        write(output/'PARENT_CLOSURE.json',gate)
        if time.monotonic()>=deadline:blocked('Four-hour deadline after closure audit');return
        freeze=output/'FULL_TASK_FREEZE.json'
        env=os.environ.copy();env.update(c.get('environment',{}));env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
        args=[c['python'],'-m','tools.task_graph.p53_full_task','prepare','--parent',plan['parent'],'--bc',str(parent/'bc_final.pt'),'--joint',str(parent/'joint_final.pt'),
            '--common-warmup',str(parent/'common.pt'),'--data',str(parent/'native_short_data.pt'),'--dino',c['dino'],'--lila',c['lila'],'--vision-python',c['visual_python'],'--vision-device','cuda','--output',str(freeze)]
        verify(plan);execute(args,env,max(1,min(1800,deadline-time.monotonic())),output,'prepare_full');verify(plan)
        full_hash=sha(freeze)
        write(output/'FULL_FREEZE_RECEIPT.json',dict(path=str(freeze),sha256=sha(freeze),launcher_plan_sha256=sha(plan_path)))
        def worker(i):
            verify(plan)
            if sha(freeze)!=full_hash:raise ValueError('Full freeze changed before cell')
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Successor deadline before launch')
            folder=output/plan['cells'][i];folder.mkdir();gpuenv=env.copy();gpuenv.update(CUDA_VISIBLE_DEVICES=str(i),MUJOCO_EGL_DEVICE_ID=str(i))
            execute([c['python'],'-m','tools.task_graph.p53_full_task','run','--binding',str(freeze),'--index',str(i),'--output',str(folder/'episode')],gpuenv,min(1800,remaining),folder,'run')
            verify(plan);result=read(folder/'episode/RESULT.json')
            if sha(folder/'episode/SEAL.json')!=result['seal_sha256']:raise ValueError('Full result seal mismatch')
            return dict(arm=plan['cells'][i],gpu=i,result=result)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,i) for i in range(3)]
            results=[];errors=[]
            for i,f in enumerate(futures):
                try:results.append(f.result())
                except Exception as e:errors.append(dict(arm=plan['cells'][i],reason=str(e),scientific_failure=False))
        write(output/'COMPLETE.json',dict(results=results,engineering_aborts=errors,expected_cells=3,automatic_retries=0,success_selection=False))
    except Exception as exc:
        write(output/'ABORT.json',dict(error=type(exc).__name__,reason=str(exc),scientific_failure=False,automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare')
    for name in ('pipeline-freeze','pipeline-output','parent','output'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--pipeline-sha',required=True);q.add_argument('--pipeline-pid',type=int,required=True)
    q=s.add_parser('run');q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':write(a.output,prepare(a.pipeline_freeze,a.pipeline_sha,a.pipeline_output,a.pipeline_pid,a.parent))
    else:run(a.plan,a.output)
