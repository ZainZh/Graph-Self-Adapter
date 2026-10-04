"""Posthoc offline action check on opened P25 harm states; no rollout/training."""
import numpy as np
import torch
import json
from tools.fast_adapter.p26_common import R,PARENT,load_adapter
from prove.fast_adapter.geometry_insert import geometry,domain
from prove.fast_adapter.insertion_teacher import action_labels
from tools.fast_adapter.p18_run import save_json,digest


def main():
    torch.set_num_threads(2)
    root=PARENT/'development/id';metrics=json.loads((root/'RESULT.json').read_text())['metrics'];rows=[];bindings=[]
    for seed in (11,22):
        rp=root/f'physical_correct_seed{seed}_RAW.pt';raw=torch.load(rp,weights_only=False)
        ids=np.flatnonzero(np.array(metrics['base']['success'])&~np.array(metrics[f'physical_correct_seed{seed}']['success']))
        v=raw['visual'][ids,:48];p=torch.tensor(raw['positions'][ids,:48],dtype=torch.float32);a=torch.tensor(raw['actions'][ids]);ba=raw['base'][ids]
        prev=torch.cat((torch.zeros_like(a[:,:1]),a[:,:-1]),1)
        lp=root/f'LATENTS_seed{seed}.pt';z=torch.load(lp,weights_only=True)['latents']['physical']['correct'][ids][:,None].expand(-1,48,-1)
        cfg=[geometry(raw['seed_base']+int(i),'id') for i in ids];dyn=[domain(raw['seed_base']+100000+int(i)) for i in ids]
        y=torch.from_numpy(action_labels(raw['positions'][ids,:48],cfg,np.array([d[0] for d in dyn]),np.array([d[1] for d in dyn])))
        y=torch.maximum(torch.minimum(y,(ba+.35).clamp(-1,1)),(ba-.35).clamp(-1,1))
        for name in ('parent','relabel','refresh'):
            net=load_adapter(seed,name,'cpu')
            with torch.no_grad():pred=net.action(v,p,prev,ba,z)
            for j,i in enumerate(ids):
                rows.append({'seed':seed,'condition':int(i),'head':name,'new_teacher_mse_last12':float((pred[j,-12:]-y[j,-12:]).square().mean()),
                    'mean_vertical_residual_last12':float((pred[j,-12:,2]-ba[j,-12:,2]).mean()),'state24_vertical_action':float(pred[j,24,2]),'state24_base_vertical':float(ba[j,24,2])})
            checkpoint=PARENT/'models'/f'seed{seed}'/'physical.pt' if name=='parent' else R/'models'/f'seed{seed}'/'final'/f'{name}.pt'
            bindings.append({'path':str(checkpoint),'sha256':digest(checkpoint)})
        bindings.extend([{'path':str(p),'sha256':digest(p)} for p in (rp,lp)])
    from pathlib import Path
    for p in [Path(__file__).relative_to(Path.cwd()),Path('prove/fast_adapter/insertion_teacher.py'),root/'RESULT.json']:
        bindings.append({'path':str(p),'sha256':digest(p)})
    save_json(R/'summary/POSTHOC_P25_HARM_ACTIONS.json',{'scope':'Posthoc offline regression on six opened P25 ID harm traces. Same frozen latent and original visual/proprio/previous-action inputs. No execution or training; action changes are not recovered successes. New teacher truth for diagnosis only.',
        'rows':rows,'source_bindings':bindings})

if __name__=='__main__':main()
