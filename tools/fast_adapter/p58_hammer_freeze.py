#!/usr/bin/env python3
"""Result-free P58 hammer training freeze."""
import argparse,hashlib,json,time
from pathlib import Path

FILES=("prove/fast_adapter/p58_hammer_observation.py",
       "prove/fast_adapter/p58_hammer_readout.py",
       "prove/fast_adapter/p58_hammer_runtime.py",
       "prove/fast_adapter/p58_geometry_model.py",
       "tools/fast_adapter/p58_hammer_export.py",
       "tools/fast_adapter/p58_hammer_readout.py",
       "tools/fast_adapter/p58_hammer_train.py")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(path,manifest=None):
    value=json.loads(Path(path).read_text())
    if value.get("schema")!="p58.hammer-training-freeze.v1":raise ValueError("Wrong P58 hammer freeze")
    for key in ("manifest","plan"):
        if sha(value[key]["path"])!=value[key]["sha256"]:raise ValueError("Changed frozen "+key)
    if manifest is not None and Path(value["manifest"]["path"]).resolve()!=Path(manifest).resolve():
        raise ValueError("Different P58 hammer manifest path")
    for source in value["source_bindings"]:
        if sha(source["path"])!=source["sha256"]:raise ValueError("Changed frozen source: "+source["path"])
    return value
def main(manifest,plan,output):
    if output.exists():raise FileExistsError(output)
    value=dict(schema="p58.hammer-training-freeze.v1",time=time.time(),result_unopened=True,
      manifest=dict(path=str(manifest),sha256=sha(manifest)),plan=dict(path=str(plan),sha256=sha(plan)),
      source_bindings=[dict(path=p,sha256=sha(p)) for p in FILES],
      readout=dict(features=263,roles=["hammer_head","target_block"],views=["global448","wrist448"],
        targets="one shared bbox-relative semantic-center UV head plus representation-specific XYZ head",
        pca_dims=[8,16,32],ridge=[.01,.1,1.,10.],
        selection=("five-fold object_id modulo 5; expert/base roles remain paired; shared UV selected by "
                   "train pixel error, each XYZ selected by .5*(pixel_mean/5 + xyz_mm_mean/5)")),
      representations=["absolute_positions","role_relative"],g_dim=6,
      runtime_canonical_geometry="head_in_current_eef_plus_target_world for both representations",
      meta=dict(seed=11,steps=500,batch=16,optimizer="Adam",lr=.0003,latent=6,
        basis="full six-dimensional PCA of train measurement residual",z0="zeros",
        inner_steps=3,inner_lr=.65,damping=.0001,step_cap=1.,geometry_scale=.01,
        target="equal-weight fixed-z0 and higher-order adapted train-only semantic position MSE",
        history=8,static_prefix="before first GT contact or >1mm target displacement; train only"),
      development_results_unopened_for_this_candidate=True,
      development_material_previously_opened=True,physics_steps=0)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps(dict(output=str(output),sha256=sha(output))),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--plan",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.manifest,a.plan,a.output)
