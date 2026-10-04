"""One train-only supervised/meta backward, no optimizer or model mutation."""
import argparse,json
from pathlib import Path
import torch
from prove.fast_adapter.p64_common import P64RoleEncoder
from prove.fast_adapter.p64_book_bridge import BookPriors
from tools.fast_adapter.p64_book_train import BookLoss,cases_for_training
from tools.fast_adapter.p64_book_data import sha
from prove.task_graph.p57_geometry_policy import load_geometry_policy


def run(cache,p61_checkpoint,lila_source,lila_checkpoint,output,device='cuda'):
    cache=Path(cache);manifest=json.loads((cache/'MANIFEST.json').read_text())
    cases=cases_for_training(manifest);case=next(c for c in cases if len(c['support_paths'])==2)
    if case['split']!='train':raise ValueError('train-only engineering')
    torch.manual_seed(11)
    model=P64RoleEncoder(lila_source,lila_checkpoint,task='book',geometry_mean=manifest['geometry_mean'],p61_checkpoint=p61_checkpoint).to(device).train()
    root=Path(__file__).resolve().parents[2]/'evaluations/fast_adapter_p57_geometry_tokens_v01'
    policy,_,_=load_geometry_policy(root/'runs_b/HEAD.pt',root/'runs_a2/mask_geometry/HEAD.pt',device=device)
    priors=BookPriors(**json.loads((cache/'PRIORS.json').read_text())['values'])
    loss=BookLoss(priors,policy);records=[]
    for phase in ('supervised','meta'):
        model.zero_grad(set_to_none=True)
        value=loss(model,case,phase);value.backward()
        audit={}
        for key,parameters in model.parameter_groups().items():
            gradients=[p.grad for p in parameters if p.grad is not None]
            audit[key]=dict(tensors=len(gradients),finite=all(torch.isfinite(g).all().item() for g in gradients),
                norm=sum(float(g.detach().square().sum()) for g in gradients)**.5)
        if not torch.isfinite(value) or any(not a['finite'] or a['norm']<=0 for a in audit.values()):raise ValueError('gradient engineering gate failed')
        records.append(dict(phase=phase,loss=float(value.detach()),groups=audit))
    result=dict(schema='prove.p64-book-gradient-engineering.v1',train_only=True,optimizer_steps=0,physics_steps=0,
        query=case['path'],support_paths=case['support_paths'],query_t=case['t'],support_times=case['meta_support_times'],
        cache_manifest_sha256=sha(cache/'MANIFEST.json'),p61_checkpoint_sha256=sha(p61_checkpoint),
        frozen_action_checkpoint_sha256=sha(root/'runs_b/HEAD.pt'),records=records,
        source_hashes={p:sha(p) for p in (__file__,'tools/fast_adapter/p64_book_train.py','prove/fast_adapter/p64_book_bridge.py','prove/fast_adapter/p64_common.py')})
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','p61-checkpoint','lila-source','lila-checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    print(json.dumps(run(a.cache,a.p61_checkpoint,a.lila_source,a.lila_checkpoint,a.output,a.device),indent=2))
