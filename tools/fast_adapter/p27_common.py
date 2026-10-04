"""Frozen-checkpoint P27 attribution; no learning or model selection."""
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p26_common import load_adapter, load_base, adaptation_hash, weight_hash
from tools.fast_adapter.p25_common import select_feedback, transitions

R=Path('evaluations/fast_adapter_p27_attribution_v01')
P26=Path('evaluations/fast_adapter_p26_actor_repair_v01')
P25=Path('evaluations/fast_adapter_p25_feedback_v01')
SEEDS=(11,22)
STRATA=('id','position','scale')
N=128
BLOCK=32
SOURCES=('correct','donor1','donor2')
SOURCE_OFFSETS=(0,400000,800000)
ARM_MODES={'parent':('correct',),'oldlabel':('correct',),'relabel':('correct',),
           'refresh':('correct','donor1','donor2','mean','matchedmean')}


def feedback(raw):
    p=torch.tensor(raw['positions'],dtype=torch.float32)
    a=torch.tensor(raw['actions']);v=raw['visual']
    idx=select_feedback(p,a)
    return idx,transitions(v,p,a,idx)


def concatenate_raw(chunks):
    result={}
    for key in ('positions','velocities','actions','visual','base'):
        xs=[x[key] for x in chunks]
        result[key]=torch.cat(xs,0) if torch.is_tensor(xs[0]) else np.concatenate(xs,axis=0)
    result.update(expert_label=None,audit_rgb_blocks=np.stack([x['audit_rgb'] for x in chunks]),
        audit_rgb_steps=chunks[0]['audit_rgb_steps'],audit_rgb_condition_offsets=list(range(0,N,BLOCK)))
    return result
