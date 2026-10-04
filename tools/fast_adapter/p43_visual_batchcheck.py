"""Diagnose BF16 batch-shape sensitivity from existing cached observations."""
import json
import numpy as np
import torch
from prove.fast_adapter.neural_geometry_visual import FrozenRoleVision
from tools.fast_adapter.p43_data import ROOT, js


def main():
    vision = FrozenRoleVision(microbatch=8)
    rows=[]
    for task in ('cup','insert'):
        pack=np.load(ROOT/'data'/task/'train/observations.npz')
        seed=int(pack['seeds'][0]);base=ROOT/'data_raw'/task/'train'/str(seed)
        raw=np.load(base/'visible_images.npz')
        rgb=raw['rgb'];masks=np.unpackbits(raw['role_masks_packed'],axis=-1)
        cached=np.load(base/'raw_visual.npy')
        vision.pca=None
        exact=vision.encode(rgb[:8],masks[:8]).cpu().numpy()
        repeated=vision.encode(np.repeat(rgb[:1],8,axis=0),np.repeat(masks[:1],8,axis=0)).cpu().numpy()[0]
        vision.microbatch=1
        single=vision.encode(rgb[:1],masks[:1]).cpu().numpy()[0]
        vision.microbatch=8
        rows.append(dict(task=task,raw_original_batch_absmax=float(np.max(np.abs(exact-cached[:8]))),
                         raw_repeated_current_frame_batch_absmax=float(np.max(np.abs(repeated-cached[0]))),
                         raw_single_absmax=float(np.max(np.abs(single-cached[0])))))
    js(ROOT/'visual/BATCH_SHAPE_DIAGNOSTIC.json',dict(records=rows));print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
