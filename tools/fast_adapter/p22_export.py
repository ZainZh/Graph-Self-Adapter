"""Export compact complete control evidence; leave dense train features in place."""
from pathlib import Path
import json,tarfile
from tools.fast_adapter.p22_data import R
from tools.fast_adapter.p18_run import save_json,digest

def main():
    assert (R/'CAMPAIGN_DONE.json').exists()
    files=[];external=[]
    for p in sorted(R.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(R)
        if p.name in ('EXPORT_RECEIPT.json','EXTERNAL_TRAIN_CACHE.json'):continue
        if rel.parts[0]=='bank' and p.name not in ('PCA.pt','TRAIN.pt','MERGE.json','RECEIPT.json','META.pt'):
            external.append({'uri':'ssh://ubuntu@100.110.12.50/home/ubuntu/github/prove-p22-geometry/'+str(p),
                             'size':p.stat().st_size,'sha256':digest(p)})
        elif p.suffix in ('.json','.pt','.jsonl','.log','.md'):
            files.append(p)
    save_json(R/'EXTERNAL_TRAIN_CACHE.json',{'assets':external,'reason':'Raw dense training features/RGB stay on execution node; all control and projected training evidence exported.'})
    files.append(R/'EXTERNAL_TRAIN_CACHE.json')
    save_json(R/'EXPORT_RECEIPT.json',{'files':[{'path':str(p),'size':p.stat().st_size,'sha256':digest(p)} for p in files]})
    files.append(R/'EXPORT_RECEIPT.json')
    with tarfile.open('/tmp/p22_results_export.tar.gz','w:gz') as t:
        for p in files:t.add(p,arcname=str(p),recursive=False)
    print(json.dumps({'files':len(files),'archive_bytes':Path('/tmp/p22_results_export.tar.gz').stat().st_size}),flush=True)

if __name__=='__main__':main()
