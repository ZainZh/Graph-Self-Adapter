"""Bounded four-GPU P18 execution; no model/agent services involved."""
import concurrent.futures as cf
import datetime, hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path

REPO=Path('/home/zheng/github/Interaction-RSI')
REL=Path('evaluations/fast_adapter_p18_visual_v01')
OUT=REPO/REL
HOST='ubuntu@100.110.12.50'
REMOTE='/home/ubuntu/github/prove-p18-visual'
LOCAL_PY='/home/zheng/miniforge3/envs/RoboTwin/bin/python3'
REMOTE_PY=REMOTE+'/runtime/bin/python'
FAMILIES=('gradient_z','history','hybrid');SEEDS=(11,22,33,44,55)
NODES=[('local',0),('remote',0),('remote',1),('remote',2)]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
    return h.hexdigest()


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')


def call(cmd,**kw):
    return subprocess.run(cmd,check=True,**kw)


def remote(command):
    return call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,command],capture_output=True,text=True).stdout


def audit_sources():
    f=json.loads((OUT/'TRAIN_FREEZE.json').read_text())
    for p,h in f['files'].items():
        assert sha(REPO/p)==h,f'local frozen source mismatch: {p}'
    for p in ('tools/fast_adapter/p18_experiment.py','tools/fast_adapter/p18_run.py','prove/fast_adapter/visual_alignment.py'):
        assert remote('sha256sum '+shlex.quote(REMOTE+'/'+p)).split()[0]==f['files'][p],p


def execute(node,family,seed,stage):
    location,gpu=node
    folder=REL/stage/family/f'seed{seed}'
    localdir=REPO/folder;localdir.mkdir(parents=True,exist_ok=True)
    if stage=='train': args=['train','--family',family,'--seed',str(seed),'--steps','1500','--batch','32','--bank',str(REL/'train_bank/bank.pt')]
    else:
        args=['eval','--domains','32' if stage=='dev' else '128','--data-seed','2026092211' if stage=='dev' else '2026092231','--checkpoint',str(REL/'train'/family/f'seed{seed}/model.pt')]
        if location=='remote':args+=['--dino',REMOTE+'/assets/dinov3']
    args+=['--output',str(folder)]
    started=time.time()
    log=localdir/'process.log'
    if location=='local':
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONPATH='.',HF_HUB_DISABLE_PROGRESS_BARS='1')
        with open(log,'w') as f:call([LOCAL_PY,'-m','tools.fast_adapter.p18_run',*args],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    else:
        command='cd '+shlex.quote(REMOTE)+' && env CUDA_VISIBLE_DEVICES='+str(gpu)+' OMP_NUM_THREADS=2 PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1 '+shlex.join([REMOTE_PY,'-m','tools.fast_adapter.p18_run',*args])
        with open(log,'w') as f:call(['ssh','-o','BatchMode=yes',HOST,command],stdout=f,stderr=subprocess.STDOUT)
        call(['rsync','-a',HOST+':'+REMOTE+'/'+str(folder)+'/',str(localdir)+'/'])
    receipt={'stage':stage,'family':family,'seed':seed,'node':node,'seconds':time.time()-started,'exit_code':0}
    (localdir/'PROCESS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return receipt


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Wait only on ordinary scripts, never keeping a child Agent alive.
    while not (OUT/'train_bank/BANK.json').exists():time.sleep(10)
    audit_sources()
    jobs=[(family,seed) for seed in SEEDS for family in FAMILIES]
    allocations={node:jobs[i::len(NODES)] for i,node in enumerate(NODES)}
    def local_train():
        return [execute(NODES[0],f,s,'train') for f,s in allocations[NODES[0]]]
    def remote_train(node):
        return [execute(node,f,s,'train') for f,s in allocations[node]]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(local_train)]
        # The local GPU can train while the shared encoded dataset transfers.
        remote('mkdir -p '+REMOTE+'/'+str(REL/'train_bank'))
        call(['rsync','-a',str(OUT/'train_bank/bank.pt'),str(OUT/'train_bank/BANK.json'),HOST+':'+REMOTE+'/'+str(REL/'train_bank')+'/'])
        assert remote('sha256sum '+REMOTE+'/'+str(REL/'train_bank/bank.pt')).split()[0]==sha(OUT/'train_bank/bank.pt')
        futures += [pool.submit(remote_train,node) for node in NODES[1:]]
        train_receipts=[x for future in futures for x in future.result()]
    write('TRAIN_RECEIPT.json',{'status':'completed','jobs':train_receipts})
    # Actual remote visual encoder must be the pinned artifact before evaluation.
    assert remote('sha256sum '+REMOTE+'/assets/dinov3/model.safetensors').split()[0]=='dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'
    # All models fixed before final data is opened. No dev-based model selection.
    checkpoints={str(REL/'train'/f/f'seed{s}/model.pt'):sha(OUT/'train'/f/f'seed{s}/model.pt') for f,s in jobs}
    write('FINAL_FREEZE.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_freeze_sha':sha(OUT/'TRAIN_FREEZE.json'),'protocol_sha':sha(OUT/'PROTOCOL.json'),'checkpoints':checkpoints,'selection':'all last-step checkpoints; no selection from development outcomes'})
    audit_sources()
    def evaluation_queue(node):
        rec=[]
        for f,s in allocations[node]:
            rec.append(execute(node,f,s,'dev'))
            rec.append(execute(node,f,s,'final'))
        return rec
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        evaluation=[x for future in [pool.submit(evaluation_queue,node) for node in NODES] for x in future.result()]
    for p,h in checkpoints.items():assert sha(REPO/p)==h,p
    write('FINAL_RECEIPT.json',{'status':'completed_all_15_final_and_15_dev','jobs':evaluation,'checkpoint_hashes_unchanged':True})
    print('P18_CAMPAIGN_COMPLETED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write('CAMPAIGN_ERROR.json',{'error':repr(e),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
