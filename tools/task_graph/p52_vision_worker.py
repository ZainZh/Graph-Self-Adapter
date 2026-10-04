#!/usr/bin/env python3
"""Single local process for frozen feature extraction; JSON lines, no network."""
import argparse
import contextlib
import json
import importlib.metadata
import sys
import numpy as np
import torch
from prove.task_graph.p52_visual_features import FrozenStageVision, public_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--dino',required=True);p.add_argument('--lila',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(1)
    with contextlib.redirect_stdout(sys.stderr):vision=FrozenStageVision(a.dino,a.lila,a.device)
    print(json.dumps(dict(ready=True,binding=vision.binding,packages={n:importlib.metadata.version(n) for n in ("torch","transformers","numpy")})),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('close'):break
        with np.load(request['public_frame'],allow_pickle=False) as frame:
            with contextlib.redirect_stdout(sys.stderr):features=public_features(frame,vision)
        # Numeric public features only; no pickle/teacher/evaluator protocol.
        print(json.dumps(dict(features={k:v.tolist() for k,v in features.items()}),allow_nan=False),flush=True)


if __name__=='__main__':main()
