"""One bounded postprocessing/export job; no agent monitoring or new rollouts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.fast_adapter.p39_tokens import ROOT,js,sha
from tools.fast_adapter.p39_launch import env

def start(module,tag,args=(),gpu=None):
    e=env(gpu) if gpu is not None else os.environ.copy()
    e.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/(tag+'.log')).open('x') as f:
        return subprocess.Popen([sys.executable,'-u','-m',module]+list(args),env=e,stdout=f,stderr=subprocess.STDOUT)

def main():
    with (ROOT/'POSTPROCESS_START.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
    deadline=time.time()+3600
    while not (ROOT/'FINAL_DONE.json').exists():
        if time.time()>deadline:raise TimeoutError('Final evaluation did not finish within bounded wait')
        time.sleep(20)
    assert json.loads((ROOT/'FINAL_DONE.json').read_text())['exits']==[0]*4
    freeze=json.loads((ROOT/'FINAL_FREEZE.json').read_text())
    for path,digest in freeze['sources_and_models'].items():assert sha(Path(path))==digest,path
    jobs=[start('tools.fast_adapter.p39_audit_v02','FINAL_AUDIT_v02',('--phase','final')),
          start('tools.fast_adapter.p39_report','FINAL_REPORT',('--phase','final')),
          start('tools.fast_adapter.p39_media_v02','FINAL_MEDIA',gpu=1)]
    codes=[p.wait() for p in jobs];assert codes==[0]*3,codes
    assert start('tools.fast_adapter.p39_display','FINAL_DISPLAY').wait()==0
    excluded=[];files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        row=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        if p.name=='FEATURES.pt':excluded.append(dict(**row,storage='ubuntu@100.110.12.50:/home/ubuntu/github/prove-p39-tokens/'+str(p)))
        else:files.append(row)
    js(ROOT/'EXPORT_MANIFEST.json',dict(files=files,remote_feature_caches=excluded,scope='Compact results, complete runtime records, training records/models and annotated videos; large frozen visual feature caches retain remote storage.'))
    archive=Path('/tmp/prove_p39_final_export.tar.gz')
    if archive.exists():raise RuntimeError('Archive already exists')
    subprocess.run(['tar','-czf',str(archive),'--exclude=*/FEATURES.pt',str(ROOT)],check=True)
    js(ROOT/'POSTPROCESS_DONE.json',dict(time=time.time(),archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(ROOT/'EXPORT_MANIFEST.json'),audit=json.loads((ROOT/'summary/final_AUDIT.json').read_text())))
    print('P39 postprocessing and export complete',flush=True)

if __name__=='__main__':main()
