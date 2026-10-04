#!/usr/bin/env python3
"""Diagnose deterministic P30 state restoration before the P58 export."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from prove.fast_adapter.held_hammer_env import HeldHammerEnv
from prove.fast_adapter.object_interaction_env import mujoco
from tools.fast_adapter.p58_hammer_export import SOURCE, restore_reset_model_parameters_without_integration


def restore(env, raw, row, t=0):
    for key, value in raw["states"].items():
        getattr(env.env.data, key)[:] = value[row, t].numpy()
    mujoco.mj_forward(env.env.model, env.env.data)


def compare(actual, expected):
    delta = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
    yy, xx = np.nonzero(np.any(delta != 0, axis=-1))
    return dict(max_pixel=int(delta.max()), changed_pixels=len(xx),
                changed_box=None if not len(xx) else [int(xx.min()), int(yy.min()),
                                                       int(xx.max()+1), int(yy.max()+1)])


def main():
    source = SOURCE/"meta_expert_000.pt"; raw=torch.load(source,map_location="cpu",weights_only=True)
    meta=json.loads(source.with_suffix(".json").read_text()); row=0; seed=int(raw["reset_seeds"][row])
    expected=raw["audit_rgb"][0,row].numpy(); result={"seed":seed,"config":meta["configs"][row]}
    direct=HeldHammerEnv(meta["configs"][row]); xy=restore_reset_model_parameters_without_integration(direct,seed)
    restore(direct,raw,row); rgb=direct.rgb(); result["direct_rng"]={**compare(rgb,expected),
        "target_xy":xy.tolist(),"target_site":direct.env.data.site("target_site").xpos.tolist()}
    full=HeldHammerEnv(meta["configs"][row]); full.reset(seed); restore(full,raw,row); rgb_full=full.rgb()
    result["held_reset"]={**compare(rgb_full,expected),
        "target_xy":full.env.model.body("target").pos[:2].tolist(),
        "target_site":full.env.data.site("target_site").xpos.tolist()}
    out=Path("/home/ubuntu/p58_hammer_run/restore_audit");out.mkdir(exist_ok=True)
    Image.fromarray(expected).save(out/"recorded.png");Image.fromarray(rgb).save(out/"direct.png")
    Image.fromarray(rgb_full).save(out/"held_reset.png")
    (out/"RESULT.json").write_text(json.dumps(result,indent=2)+"\n")
    direct.close();full.close();print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
