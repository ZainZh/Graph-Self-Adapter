"""Pause only old dispatcher; adopt its still-running jobs, then share pending jobs."""
import os,signal,re,threading,time,json,datetime,subprocess,concurrent.futures as cf
from tools.fast_adapter.p18_campaign import *

OLD_PID=2115956

def pause_and_capture():
    cmd=Path(f'/proc/{OLD_PID}/cmdline').read_bytes().split(b'\0')
    assert b'tools/fast_adapter/p18_campaign.py' in cmd
    for attempt in range(5):
        os.kill(OLD_PID,signal.SIGSTOP)
        lines=subprocess.check_output(['ps','--ppid',str(OLD_PID),'-o','pid=,args='],text=True).splitlines()
        active=[];valid=True
        for line in lines:
            match=re.search(r'--output '+str(REL)+r'/(dev|final)/([^/]+)/seed(\d+)',line)
            gpu=re.search(r'CUDA_VISIBLE_DEVICES=(\d+)',line)
            if not match or not gpu or 'ssh -o BatchMode=yes '+HOST not in line:valid=False;break
            active.append({'pid':int(line.split()[0]),'stage':match[1],'family':match[2],'seed':int(match[3]),'node':['remote',int(gpu[1])]})
        if valid and len(active)==3:return active
        os.kill(OLD_PID,signal.SIGCONT);time.sleep(5)
    raise RuntimeError('dispatcher handoff was not at a safely identifiable boundary')


def adopt(job):
    stage,family,seed=job['stage'],job['family'],job['seed'];folder=REL/stage/family/f'seed{seed}'
    result=REMOTE+'/'+str(folder)+'/EVAL.json';start=time.time()
    while True:
        ready=remote('test -f '+shlex.quote(result)+' && echo ready || true').strip()
        if ready=='ready':break
        if time.time()-start>1800:raise RuntimeError('adoption timeout '+str(job))
        time.sleep(15)
    local=REPO/folder;local.mkdir(parents=True,exist_ok=True)
    call(['rsync','-a',HOST+':'+REMOTE+'/'+str(folder)+'/',str(local)+'/'])
    rec=json.loads((local/'EVAL.json').read_text());assert rec['all_model_parameters_unchanged']
    assert rec['raw_sha256']==sha(local/'raw.jsonl')
    receipt={**job,'status':'completed_from_bound_EVAL_receipt','adopted_without_stopping_evaluation':True}
    (local/'PROCESS_ADOPTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if stage=='dev' and not (OUT/'final'/family/f'seed{seed}/EVAL.json').exists():execute(tuple(job['node']),family,seed,'final')


def main():
    assert (OUT/'FINAL_FREEZE.json').exists()
    active=pause_and_capture()
    occupied={(j['family'],j['seed']) for j in active}
    pending=[(f,s) for s in SEEDS for f in FAMILIES if (f,s) not in occupied and not (OUT/'final'/f/f'seed{s}/EVAL.json').exists()]
    write('SCHEDULING_HANDOFF.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_dispatcher_pid':OLD_PID,'old_dispatcher_stopped_not_evaluation_processes':True,'active_jobs_continuing':active,'remaining_models':pending,'reason':'local GPU finished its static allocation; use idle GPU and next-free worker','scientific_changes':'none: same sealed source, checkpoint, all test domains, queries, K and controls; no outcome-based selection'})
    lock=threading.Lock()
    def worker(node):
        for job in active:
            if tuple(job['node'])==node:adopt(job)
        while True:
            with lock:
                if not pending:return
                family,seed=pending.pop(0)
            if not (OUT/'dev'/family/f'seed{seed}/EVAL.json').exists():execute(node,family,seed,'dev')
            if not (OUT/'final'/family/f'seed{seed}/EVAL.json').exists():execute(node,family,seed,'final')
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,node) for node in NODES]
        for future in futures:future.result()
    audit_sources()
    freeze=json.loads((OUT/'FINAL_FREEZE.json').read_text())
    for p,h in freeze['checkpoints'].items():assert sha(REPO/p)==h
    jobs=[]
    for stage in ('dev','final'):
        for f in FAMILIES:
            for s in SEEDS:
                p=OUT/stage/f/f'seed{s}';rec=json.loads((p/'EVAL.json').read_text())
                assert rec['raw_sha256']==sha(p/'raw.jsonl') and rec['all_model_parameters_unchanged']
                jobs.append({'stage':stage,'family':f,'seed':s,'receipt':str(p.relative_to(REPO)/'EVAL.json'),'receipt_sha256':sha(p/'EVAL.json'),'seconds':rec['seconds']})
    write('FINAL_RECEIPT.json',{'status':'completed_all_15_final_and_15_dev','jobs':jobs,'checkpoint_hashes_unchanged':True,'scheduling_handoff':'SCHEDULING_HANDOFF.json'})
    # All adopted evaluation children have completed; release old stopped parent.
    os.kill(OLD_PID,signal.SIGTERM);os.kill(OLD_PID,signal.SIGCONT)
    print('P18_CAMPAIGN_COMPLETED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write('RESUME_ERROR.json',{'error':repr(e),'old_dispatcher_remains_stopped_for_safe_recovery':True})
        raise
