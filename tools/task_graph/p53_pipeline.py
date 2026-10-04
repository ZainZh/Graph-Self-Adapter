#!/usr/bin/env python3
"""Finite P53 post-collection pipeline. Prepare does not train or run physics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import signal
import time
from tools.task_graph.p53_dataset import catalog,read,write
from tools.task_graph.p52_pack_dataset import sha
from tools.task_graph.p52_deployment import source_closure

ROOT=Path(__file__).resolve().parents[2]


def kill_process_tree(pid):
    """Stop descendants first, including queue workers in separate sessions."""
    stopped={pid}
    try:os.kill(pid,signal.SIGSTOP)
    except ProcessLookupError:pass
    while True:
        found=set()
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields=path.read_text().rsplit(')',1)[1].split()
                child=int(path.parent.name)
                if int(fields[1]) in stopped and child not in stopped:found.add(child)
            except (OSError,ValueError,IndexError):continue
        if not found:break
        for child in found:
            try:os.kill(child,signal.SIGSTOP)
            except ProcessLookupError:pass
        stopped.update(found)
    for child in stopped:
        try:
            if os.getpgid(child)==child:os.killpg(child,signal.SIGKILL)
            else:os.kill(child,signal.SIGKILL)
        except ProcessLookupError:pass


def execute(argv,env,timeout,output,name):
    started=time.time();proc=None
    receipt=dict(argv=argv,started=started,automatic_retry=False,returncode=None)
    with (output/(name+'.log')).open('x') as log:
        try:
            proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt.update(returncode=proc.wait(timeout=timeout),execution_status='finished')
        except subprocess.TimeoutExpired:
            kill_process_tree(proc.pid);proc.wait()
            receipt.update(execution_status='engineering_timeout',system_success=None)
        except OSError as error:
            receipt.update(execution_status='launch_error',error=str(error),system_success=None)
    receipt['finished']=time.time();write(output/(name+'.receipt.json'),receipt)
    if receipt['returncode']!=0:
        write(output/(name+'.ABORT.json'),receipt)
        raise RuntimeError('Pipeline command failed without retry: '+name)
    return receipt


def prepare(config):
    report=catalog(config['manifest'])
    if report['cohort_size']!=24 or report['split_counts']!={'train':18,'validation':6}:raise ValueError('Original24 and18/6 required')
    if any(r['status']!='sealed' for r in report['original_collection']):raise ValueError('All24 collection results must be closed and sealed')
    if config['min_qualified_train']!=12:raise ValueError('Root-approved minimum is12 qualified train, with all6 grasp groups')
    if not 1800<=config['command_timeout_seconds']<=14400:raise ValueError('Explicit bounded preprocessing/training timeout required')
    receipts=config['collection_receipts']
    if not receipts or sum(read(p)['expected_jobs'] for p in receipts)!=24:raise ValueError('Closed finite24 collection queue receipts required')
    attempts=[r for p in receipts for r in read(p)['attempts']]
    if len(attempts)!=24 or len({r['job']['id'] for r in attempts})!=24 or any(r['execution_status']!='finished' or r['returncode']!=0 for r in attempts):raise ValueError('Collection queue incomplete or engineering abort')
    for key in ('python','visual_python'):
        if not Path(config[key]).is_file():raise ValueError('Explicit existing runtime required')
    files=[config['manifest'],report['collection_binding']['path'],*receipts,*config['p52_checkpoints'].values()]
    for row in report['original_collection']+report['historical_qualifications']:
        files.extend([row['source_result'],row['source_seal'],str(Path(row['directory'])/'BINDING.json')])
    files += [str(Path(config['dino'])/p) for p in ('model.safetensors','config.json','preprocessor_config.json')]
    files += [str(Path(config['lila'])/p) for p in ('VISUAL_WEIGHTS.pt','models/vla_model_fm.py')]
    entries=['tools/task_graph/p53_pipeline.py','tools/task_graph/p53_dataset.py','tools/task_graph/p53_train_short.py','tools/task_graph/p53_micro_evaluate.py','tools/task_graph/p52_vision_worker.py','tools/task_graph/p53_queue.py']
    return dict(schema='p53.finite-pipeline.v1',config=config,catalog=report,sources={p:sha(ROOT/p) for p in source_closure(ROOT,entries)},
        inputs={str(Path(p).resolve()):sha(p) for p in files},scope='opened development, final checkpoints only, no retries or replacements',
        budget=dict(feature_workers=3,common=1000,branch=2000,seed=11,eval_cells=36,horizon=24,min_qualified_train=12,required_grasp_groups=6,eval_workers=3),automatic_retries=0)


def verify(plan):
    if plan['schema']!='p53.finite-pipeline.v1' or plan['automatic_retries']!=0:raise ValueError('Wrong pipeline contract')
    for p,h in plan['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen source changed:'+p)
    for p,h in plan['inputs'].items():
        if sha(p)!=h:raise ValueError('Bound input changed:'+p)


def run(plan_path,output):
    plan=read(plan_path);verify(plan);c=plan['config'];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/'PLAN.json',plan);write(output/'CATALOG.json',catalog(c['manifest']))
    conditions={e['job_id']:e for e in read(plan['catalog']['collection_binding']['path'])['conditions']}
    groups={conditions[r['episode_id']]['source_condition']['grasp_index'] for r in plan['catalog']['original_collection'] if r['split']=='train' and r['eligible_bc']}
    if plan['catalog']['eligible_counts']['train']<12 or groups!=set(range(6)):
        write(output/'STOP.json',dict(reason='Insufficient predeclared training trajectories',automatic_retry=False));return
    def command(name,args,gpu=0,visual=False):
        verify(plan);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        env.update(c.get('environment',{}));argv=[c['visual_python'] if visual else c['python'],'-m',*map(str,args)]
        execute(argv,env,c['command_timeout_seconds'],output,name)
        verify(plan)
    features=output/'features';data=output/'native_short_data.pt';common=output/'common.pt';bc=output/'bc_final.pt';joint=output/'joint_final.pt'
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(command,f'features_{i}',['tools.task_graph.p53_dataset','extract','--manifest',c['manifest'],'--feature-dir',features,'--dino',c['dino'],'--lila',c['lila'],'--device','cuda','--shards','3','--shard-index',i],i,True) for i in range(3)]
        for f in futures:f.result()
    command('pack',['tools.task_graph.p53_dataset','pack','--manifest',c['manifest'],'--feature-dir',features,'--output',data])
    command('warmup',['tools.task_graph.p53_train_short','warmup','--data',data,'--output',common,'--horizon','24','--device','cuda'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(command,branch,['tools.task_graph.p53_train_short','branch','--data',data,'--common',common,'--output',checkpoint,'--branch',branch,'--device','cuda'],gpu) for branch,checkpoint,gpu in [('joint',joint,0),('bc',bc,1)]]
        for f in futures:f.result()
    write(output/'FINAL_WEIGHTS.json',{str(p):sha(p) for p in (data,common,bc,joint)})
    rows=[r for r in plan['catalog']['original_collection'] if r['split']=='validation'];entries=[]
    for row in rows:
        entry={k:row[k] for k in ('near_entry_qualified','source_result','source_result_sha256','source_seal','source_seal_sha256')}
        entry.update(name=row['episode_id'],split='validation')
        if row['near_entry_qualified']:entry.update(frame=row['entry_frame'],frame_sha256=row['entry_frame_sha256'])
        entries.append(entry)
    entry_path=output/'ENTRIES.json';write(entry_path,entries);freeze=output/'MICRO_FREEZE.json'
    command('eval_prepare',['tools.task_graph.p53_micro_evaluate','prepare','--parent',plan['catalog']['collection_binding']['path'],'--entries',entry_path,
        '--p52-bc',c['p52_checkpoints']['bc'],'--p52-joint',c['p52_checkpoints']['joint'],'--p53-bc',bc,'--p53-joint',joint,
        '--dino',c['dino'],'--lila',c['lila'],'--vision-python',c['visual_python'],'--output',freeze])
    write(output/'EVAL_FREEZE_RECEIPT.json',dict(path=str(freeze),sha256=sha(freeze),plan_sha256=sha(plan_path)))
    # Exactly three finite workers, one simulator/visual pair per physical GPU.
    queue=dict(schema='p53.explicit-queue.v1',workers=3,attempts_per_job=1,cwd=str(ROOT),
        bound_files={**{str(ROOT/p):h for p,h in plan['sources'].items()},str(freeze):sha(freeze),**read(output/'FINAL_WEIGHTS.json')},
        jobs=[dict(id=f'cell_{i:02d}',seed=11,timeout_seconds=1800,argv=[c['python'],'-m','tools.task_graph.p53_micro_evaluate','run','--binding',str(freeze),'--index',str(i)]) for i in range(36)])
    queue_path=output/'EVAL_QUEUE.json';write(queue_path,queue);verify(plan)
    # Queue owns CUDA_VISIBLE_DEVICES for each of its workers; do not mask here.
    env=os.environ.copy();env.update(c.get('environment',{}));started=time.time()
    argv=[c['python'],'-m','tools.task_graph.p53_queue','--manifest',str(queue_path),'--manifest-sha256',sha(queue_path),'--output',str(output/'eval')]
    execute(argv,env,36*1800+600,output,'eval_queue')
    closed=read(output/'eval/QUEUE_RESULT.json');results=[]
    if len(closed['attempts'])!=36:raise ValueError('Incomplete evaluation queue')
    for receipt in closed['attempts']:
        cell=output/'eval'/receipt['job']['id']/'episode/RESULT.json'
        results.append(read(cell) if cell.is_file() else dict(cell=receipt['job']['id'],engineering_abort=receipt['execution_status'],system_success=None))
    verify(plan);write(output/'COMPLETE.json',dict(results=results,expected_cells=36,closed_queue_attempts=len(closed['attempts']),sealed_scientific_results=sum(r.get('system_success') is not None for r in results),engineering_aborts=sum(r.get('system_success') is None for r in results),automatic_retries=0,all_original_entries_retained=True))



if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--config',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('run');q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':write(a.output,prepare(read(a.config)))
    else:run(a.plan,a.output)
