"""One actual training-cache episode: exact frozen loss, no optimizer or physics.

Temporarily replace only the driver's common trainer binding with an audit
callback. This invokes the existing nested loss verbatim, then restores the
binding in finally. The formal trainer, its freeze checks and budgets are not
modified; no checkpoint is produced by this diagnostic.
"""
import argparse
import json
from pathlib import Path
import random
import time
import torch

from prove.fast_adapter.p61_data import sha256
from prove.fast_adapter.p60_contour import model_state_sha256


def run(cache,initial_checkpoint,action_checkpoint,output,device='cuda'):
    import prove.fast_adapter.p64_common as common
    from tools.fast_adapter.p64_cup_train import train
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    records=[]
    def audit(model,cases,loss_callback,**kwargs):
        model.to(device).train()
        before=model_state_sha256(model)
        case=cases[0]
        for phase in ('supervised','meta'):
            random.seed(11);torch.manual_seed(11)
            model.zero_grad(set_to_none=True)
            started=time.monotonic()
            loss=loss_callback(model,case,phase)
            if loss.ndim or not torch.isfinite(loss):raise FloatingPointError('Non-finite actual '+phase+' loss')
            loss.backward()
            groups={}
            for name,module in [('official_fusion',model.backbone.fusion),
                                ('official_query_adapter',model.backbone.visual_adapter),
                                ('new_task_head',model.geometry_head)]:
                grads=[p.grad.detach() for p in module.parameters() if p.grad is not None]
                finite=all(bool(torch.isfinite(g).all()) for g in grads)
                norm=sum(float(g.double().square().sum()) for g in grads)**.5
                groups[name]=dict(parameters_with_gradient=len(grads),finite=finite,norm=norm)
                if not grads or not finite or norm<=0:raise RuntimeError('Invalid '+phase+' gradient '+name)
            all_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if not all_finite:raise FloatingPointError('Non-finite task gradients')
            records.append(dict(phase=phase,loss=float(loss.detach()),groups=groups,all_gradients_finite=all_finite,
                elapsed_seconds=time.monotonic()-started,optimizer_updates=0))
            print(json.dumps(records[-1]),flush=True)
        after=model_state_sha256(model)
        if before!=after:raise RuntimeError('Weights changed in gradient-only check')
        return dict(weights_before=before,weights_after=after,weights_unchanged=True,
                    case=case,records=records)
    original=common.train_encoder
    common.train_encoder=audit
    try:
        result=train(cache,str(output)+'.NEVER_WRITTEN.pt',initial_checkpoint,action_checkpoint,
                     'NO_FORMAL_FREEZE_NO_OPTIMIZER',device)
    finally:
        common.train_encoder=original
    result.update(schema='prove.p64-cup-exact-loss-gradient-check.v1',optimizer_updates=0,physics_steps=0,
        driver_sha256=sha256('tools/fast_adapter/p64_cup_train.py'),
        cache_manifest_sha256=sha256(Path(cache)/'MANIFEST.json'),
        initial_checkpoint_sha256=sha256(initial_checkpoint),action_checkpoint_sha256=sha256(action_checkpoint),
        method='Exact nested driver callback via temporary trainer injection; restored after audit; no optimizer constructed')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--initial-checkpoint',required=True)
    p.add_argument('--action-checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(a.cache,a.initial_checkpoint,a.action_checkpoint,a.output,a.device)
