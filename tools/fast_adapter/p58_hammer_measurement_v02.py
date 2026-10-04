#!/usr/bin/env python3
"""Derive P58 hammer V2 public measurements from the frozen V1 export."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from prove.fast_adapter.p58_hammer_observation_v02 import (
    PUBLIC_KEYS, clean_role_masks, public_frame, semantic_uv_valid,
)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+"\n")

def derive(source_manifest,output):
    source_manifest=Path(source_manifest);output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);source=json.loads(source_manifest.read_text());root=source_manifest.parent
    rows=[];removed=[];label_valid=[]
    for item in source["rows"]:
        old_public=root/item["public_path"];old_private=root/item["private_path"]
        if sha(old_public)!=item["public_sha256"] or sha(old_private)!=item["private_sha256"]:
            raise ValueError("V1 source NPZ hash mismatch")
        with np.load(old_public,allow_pickle=False) as blob:
            public={key:blob[key] for key in blob.files}
        with np.load(old_private,allow_pickle=False) as blob:
            private={key:blob[key] for key in blob.files}
        values={key:[] for key in PUBLIC_KEYS}
        for index in range(len(public["times"])):
            raw=np.asarray(public["masks"][index],np.uint8);clean=clean_role_masks(raw)
            encoded=public_frame(clean,public["camera_K"][index],public["camera_pose"][index],
                public["proprio"][index],public["eef_transform"][index])
            encoded.update(rgb=public["rgb"][index],masks=clean)
            if set(encoded)!=PUBLIC_KEYS:raise ValueError("V2 public allowlist changed")
            for key in PUBLIC_KEYS:values[key].append(encoded[key])
            removed.append((raw.reshape(2,2,-1).sum(-1)-clean.reshape(2,2,-1).sum(-1)).astype(int))
            label_valid.append(semantic_uv_valid(private["semantic_uv"][index],
                private["semantic_depth"][index],encoded["mask_valid"]))
        public_name=item["public_path"];private_name=item["private_path"]
        np.savez_compressed(output/public_name,times=public["times"],
            **{key:np.stack(values[key]) for key in sorted(PUBLIC_KEYS)})
        shutil.copy2(old_private,output/private_name)
        if sha(output/private_name)!=item["private_sha256"]:raise ValueError("Private bytes changed")
        row=dict(item);row.update(public_sha256=sha(output/public_name),private_sha256=sha(output/private_name))
        rows.append(row)
    removed=np.stack(removed);valid=np.stack(label_valid)
    manifest=dict(source)
    manifest.update(schema="p58.hammer-geometry-export.v2",rows=rows,
        derived_without_render_or_physics=True,parent_manifest=dict(path=str(source_manifest),sha256=sha(source_manifest)),
        measurement_revision=dict(mask_cleanup="largest 8-connected component; equal-size tie uses earliest row-major component",
          uv_training_valid="clean role visible and positive depth and semantic UV in [0,448)^2",
          runtime_valid="clean role visible and predicted UV in [0,448)^2; no evaluator truth",
          removed_pixel_quantiles=np.quantile(removed,[0,.5,.9,.95,.99,1]).tolist(),
          uv_valid_counts=valid.sum(0).tolist(),uv_total_per_view_role=int(valid.shape[0])),
        sources=source["sources"]+[dict(path="prove/fast_adapter/p58_hammer_observation_v02.py",sha256=sha("prove/fast_adapter/p58_hammer_observation_v02.py")),dict(path=str(Path(__file__)),sha256=sha(Path(__file__)))])
    write(output/"MANIFEST.json",manifest)
    print(json.dumps(dict(output=str(output),manifest_sha256=sha(output/"MANIFEST.json"),
      removed_pixel_quantiles=manifest["measurement_revision"]["removed_pixel_quantiles"],
      uv_valid_counts=manifest["measurement_revision"]["uv_valid_counts"])),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();derive(a.source_manifest,a.output)
