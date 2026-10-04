"""Export sealed control evidence; keep large dense features on the execution node."""
from pathlib import Path
import json,tarfile,shutil
import torch
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p21_frontends_v01')
def main():
    assert json.loads((R/'FINAL_BATCH_DONE.json').read_text())['completed']
    files=[];remote=[]
    for p in sorted(R.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(R)
        if rel.parts[0]=='bank':
            remote.append({'uri':'ssh://ubuntu@100.110.12.50/home/ubuntu/github/prove-p21-frontends/'+str(p),'size':p.stat().st_size,'sha256':digest(p)})
            if p.name in ['RECEIPT.json','LOCAL_SOURCE_RECEIPT.json','META.pt']:
                copy=R/'REMOTE_CACHE_METADATA'/rel.parts[1]/p.name;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,copy);files.append(copy)
        elif p.name=='LATENTS_BEFORE_QUERY.pt':
            state=torch.load(p,map_location='cpu',weights_only=False)
            state.pop('support_raw_dino',None)
            out=p.with_name('LATENTS_COMPACT.pt');torch.save(state,out);files.append(out)
            remote.append({'uri':'ssh://ubuntu@100.110.12.50/home/ubuntu/github/prove-p21-frontends/'+str(p),'size':p.stat().st_size,'sha256':digest(p),'compact_export':str(out),'compact_sha256':digest(out),'omitted':'support_raw_dino only; support_visual, all z, curves, seeds, model hashes retained'})
        elif p.suffix in ['.json','.pt','.jsonl']:
            files.append(p)
    save_json(R/'EXTERNAL_FEATURE_CACHE.json',{'remote_artifacts':remote,'reason':'lossless RGB/dense visual caches retained on trusted execution node; compact control evidence returned'})
    files.append(R/'EXTERNAL_FEATURE_CACHE.json')
    save_json(R/'EXPORT_RECEIPT.json',{'files':[{'path':str(p),'sha256':digest(p),'size':p.stat().st_size} for p in files],'full_control_raw_exported':True,'latent_raw_dino_only_retained_remote':True})
    files.append(R/'EXPORT_RECEIPT.json')
    with tarfile.open('/tmp/p21_results_export.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p),recursive=False)
    print(json.dumps({'archive':'/tmp/p21_results_export.tar.gz','size':Path('/tmp/p21_results_export.tar.gz').stat().st_size,'files':len(files)}),flush=True)
if __name__=='__main__':main()
