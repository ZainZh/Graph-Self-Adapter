#!/usr/bin/env python3
"""Fixed offline query action-error comparison, not a physical success evaluator."""
import argparse
import json
from pathlib import Path
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy
from tools.task_graph.p52_train import frame,validate


def evaluate(model,episodes,split):
    validate(episodes);rows=[]
    model.eval().requires_grad_(False)
    for e in episodes:
        if e['split']!=split:continue
        f=e['features'];o=frame(f,0)
        z0=model.initial(o).detach();z=z0.clone();fixed=[];online=[];history=[]
        for t in range(len(e['executed_action'])):
            o=frame(f,t);label=e['expert_action'][t]
            with torch.no_grad():
                fixed.append(float((model.action(o,z0)-label).square().mean()))
                online.append(float((model.action(o,z)-label).square().mean()))
            history.append((o,e['executed_action'][t],frame(f,t+1)));history=history[-model.config.history:]
            z,_=model.adapt_history(history,z,z0)
            z=z.detach()
        rows.append(dict(episode_id=e['episode_id'],transitions=len(fixed),fixed_mse=sum(fixed)/len(fixed),
                         online_mse=sum(online)/len(online)))
    return dict(claim='Recorded-trajectory offline action error only; not closed-loop success',
                split=split,episodes=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise FileExistsError(a.output)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    m=StagePolicy(Config(**ck['config']));m.load_state_dict(ck['model'],strict=True)
    result=evaluate(m,torch.load(a.data,map_location='cpu',weights_only=True)['episodes'],a.split)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
