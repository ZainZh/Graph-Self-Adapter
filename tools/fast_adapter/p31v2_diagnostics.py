"""Post-training diagnostics, explicitly separate from main controlled results."""
import torch
import numpy as np
from tools.fast_adapter.p31v2_common import *
from tools.fast_adapter.p31v2_data import relation_labels
from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco


def training_feasibility():
    torch.set_num_threads(2);base=base_model(TASK)
    feasibility=[]
    for kind in ('old','own'):
        b={k:v.cuda() for k,v in torch.load(ROOT/'data'/f'{kind}.pt',weights_only=True).items()}
        previous=torch.cat((torch.zeros_like(b['actions'][:,:1]),b['actions'][:,:-1]),1)
        with torch.no_grad():ba=base(b['visual'][:,:-1],b['proprio'][:,:-1],previous)
        lo=(ba-.5).clamp(-1,1);hi=(ba+.5).clamp(-1,1);teacher=b['teacher']
        best=torch.maximum(lo,torch.minimum(hi,teacher))
        near=b['near'];close=(teacher[...,3]>.5)&near
        feasibility.append({'bank':kind,'teacher_frames':int(close.sum()),
             'near_close_teacher_unreachable_fraction':float((teacher[...,3][close]>hi[...,3][close]).float().mean()),
             'near_close_base_cannot_even_reverse_to_positive_fraction':float((hi[...,3][close]<=0).float().mean()),
             'base_teacher_action_mse':float((ba-teacher).square().mean()),
             'bounded_residual_minimum_possible_action_mse':float((best-teacher).square().mean())})
    return feasibility


def main():
    feasibility=training_feasibility()
    # The frozen stage observer is also scored on actually visited development
    # states only AFTER they were sealed. These labels cannot change candidates.
    groups={}
    for seed in (11,22):
        result=json.loads((ROOT/'development'/f'seed{seed}'/'RESULT.json').read_text())
        for row in result['rows']:groups.setdefault(row['source']['path'],[]).append(row)
    gate_counts={}
    for path,rows in groups.items():
        b=torch.load(path,weights_only=True);key=rows[0]['split']+'/'+rows[0]['arm']
        counts=gate_counts.setdefault(key,dict(tp=0,tn=0,fp=0,fn=0,positive_frames=0,negative_frames=0))
        for row in rows:
            j=row['source_row'];env=ObjectEnv(row['config'],render=False);env.reset(row['reset_seed'])
            try:
                for t in range(200):
                    for k,ss in b['states'].items():getattr(env.env.data,k)[:]=ss[j,t].numpy()
                    mujoco.mj_forward(env.env.model,env.env.data)
                    _,truth=relation_labels(env);pred=bool(b['active'][j,t])
                    label=('tp' if pred else 'fn') if truth else ('fp' if pred else 'tn')
                    counts[label]+=1;counts['positive_frames' if truth else 'negative_frames']+=1
            finally:env.close()
    for c in gate_counts.values():
        c['recall']=c['tp']/max(1,c['positive_frames']);c['specificity']=c['tn']/max(1,c['negative_frames'])
    dump(ROOT/'summary/DIAGNOSTICS.json',{'scope':'posthoc training feasibility and sealed development observer diagnostic; no changes to candidates',
         'training_action_feasibility':feasibility,'development_gate_quality':gate_counts})
    print('DIAGNOSTICS_COMPLETE',feasibility,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--training-only',action='store_true');a=p.parse_args()
    if a.training_only:
        rows=training_feasibility();dump(ROOT/'TRAINING_FEASIBILITY.json',{'posthoc_training_diagnostic_only':True,'rows':rows});print(rows)
    else:main()
