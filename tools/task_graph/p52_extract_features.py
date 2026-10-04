#!/usr/bin/env python3
"""Extract actual frozen pretrained features from explicit public frame files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from prove.task_graph.p52_visual_features import FrozenStageVision, public_features
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frames',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dino',default=DEFAULT_MODEL)
    p.add_argument('--lila',default='data/sources/lila-wam-reference-20260923')
    p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError(args.output)
    vision=FrozenStageVision(args.dino,args.lila,args.device)
    rows=[];sources=[]
    for path in args.frames:
        with np.load(path,allow_pickle=False) as f:
            rows.append(public_features(f,vision))
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out={k:torch.as_tensor(np.stack([r[k] for r in rows])) for k in rows[0]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='p52_public_features_v01',features=out,sources=sources,
                    frontend=vision.binding),args.output)
    print(json.dumps(dict(frames=len(rows),shapes={k:list(v.shape) for k,v in out.items()},
                          frontend=vision.binding,output=str(args.output)),indent=2))


if __name__=='__main__':main()
