#!/usr/bin/env python3
"""Build one remote-only path derivative for shared inventories/catalogs/XML."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASKS = (
    "collect_apple", "collect_cubes", "hanging_mug", "open_drawer_and_place_object",
    "put_knife_board_new", "put_mug_on_box", "sort_two_spoon",
    "sorting_cube_bimanual", "stack_cups", "sweep_objects",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rewrite(value, replacements):
    if isinstance(value, dict): return {rewrite(key, replacements): rewrite(item, replacements) for key, item in value.items()}
    if isinstance(value, list): return [rewrite(item, replacements) for item in value]
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mujoco", type=Path, required=True)
    parser.add_argument("--robort", type=Path, required=True)
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    repo=args.repo.resolve();mujoco=args.mujoco.resolve()
    replacements = (
        ("/home/zheng/github/Graph-Self-Adapter", str(repo)),
        ("/home/zheng/github/Mujoco-RSI", str(mujoco)),
        ("/home/zheng/github/RoboTwin_bin_pick", str(args.robort.resolve())),
        ("/home/zheng/github/SAM2_streaming", str(args.sam.resolve())),
        ("/home/zheng/Datasets/sam2/sam2.1_hiera_small.pt", str(args.checkpoint.resolve())),
    )
    owner=repo/"evaluations/multitask_parallel_v22/server_shared";owner.mkdir(parents=True,exist_ok=True)
    inputs = {
        "TASK_ASSET_INVENTORY_SERVER.json": repo/"evaluations/multitask_runtime_v20/TASK_ASSET_INVENTORY.json",
        "DUAL_TOOL_INVENTORY_SERVER.json": repo/"evaluations/multitask_parallel_v22/dual_tool/TASK_ASSET_INVENTORY.json",
        "EXTERNAL_ASSET_BINDINGS_V01_SERVER.json": repo/"evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json",
        "EXTERNAL_ASSET_BINDINGS_V02_SERVER.json": repo/"evaluations/twin_rsi_v02/EXTERNAL_ASSET_BINDINGS.json",
    }
    records=[]
    for name,source in inputs.items():
        target=owner/name;source_sha=digest(source);value=rewrite(json.loads(source.read_text()),replacements)
        target.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
        if "/home/zheng/" in target.read_text(): raise RuntimeError("unmapped local path in "+str(target))
        records.append({"kind":"json_derivative","source":str(source),"source_sha256":source_sha,"target":str(target),"target_sha256":digest(target)})
    for task in TASKS:
        scene=mujoco/"runs/physics_v2"/task/"scene.xml"
        source_bytes=scene.read_bytes();source_sha=hashlib.sha256(source_bytes).hexdigest();text=source_bytes.decode()
        if "/home/zheng/github/Mujoco-RSI" not in text: raise RuntimeError("scene source was already modified: "+str(scene))
        target_text=rewrite(text,replacements)
        if "/home/zheng/" in target_text: raise RuntimeError("unmapped local scene path: "+str(scene))
        scene.write_text(target_text)
        records.append({"kind":"in_place_remote_scene_derivative","source_path_on_origin":f"/home/zheng/github/Mujoco-RSI/runs/physics_v2/{task}/scene.xml","source_sha256":source_sha,"target":str(scene),"target_sha256":digest(scene)})
    manifest={"schema":"graph-self-adapter.remote-shared-derivatives.v1","server":"sunzheng",
        "isolated_remote_only":True,"historical_sources_modified":False,"replacements":[{"source":a,"target":b} for a,b in replacements],
        "records":records,"owner":"single shared deployment owner","task_count":len(TASKS)}
    path=owner/"REMOTE_DERIVATIVE_MANIFEST.json";path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"manifest":str(path),"sha256":digest(path),"records":len(records)}))


if __name__ == "__main__": main()
