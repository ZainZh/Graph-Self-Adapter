#!/usr/bin/env python3
"""Derive only drawer/hanging server configs; shared catalogs are another owner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GRAPH = Path("/home/ubuntu/github/Graph-Self-Adapter-v22")
MAPPINGS = (
    ("/home/zheng/github/xtrainer_clover/dobot_control/robots/dobot_api.py", "/home/ubuntu/github/RoboTwin_bin_pick/include/xtrainer_clover/dobot_control/robots/dobot_api.py"),
    ("/home/zheng/github/Graph-Self-Adapter", str(GRAPH)),
    ("/home/zheng/github/Mujoco-RSI", "/home/ubuntu/github/Mujoco-RSI-v22"),
    ("/home/zheng/github/RoboTwin_bin_pick", "/home/ubuntu/github/RoboTwin_bin_pick"),
    ("/home/zheng/github/SAM2_streaming", "/home/ubuntu/github/SAM2_streaming"),
    ("/home/zheng/Datasets/sam2/sam2.1_hiera_small.pt", "/home/ubuntu/Datasets/sam2/sam2.1_hiera_small.pt"),
    ("/home/zheng/.config/graph-self-adapter/openrouter.key", "/home/ubuntu/.config/graph-self-adapter/openrouter.key"),
)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rewrite(value):
    if isinstance(value, str):
        for source, target in MAPPINGS: value=value.replace(source,target)
        return value
    if isinstance(value, list): return [rewrite(item) for item in value]
    if isinstance(value, dict): return {key:rewrite(item) for key,item in value.items()}
    return value

def main():
    records=[]
    for task in ("hanging_mug","open_drawer_and_place_object"):
        source=GRAPH/"evaluations/multitask_runtime_v20"/task/"CONFIG_PROPOSED.json"
        target=GRAPH/"evaluations/multitask_parallel_v22/articulated"/task/"server_fair_v22_3/CONFIG_SERVER.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(rewrite(json.loads(source.read_text())),indent=2,sort_keys=True)+"\n")
        records.append({"task":task,"source_path":str(source),"source_sha256":sha(source),"derivative_path":str(target),"derivative_sha256":sha(target)})
    manifest=GRAPH/"evaluations/multitask_parallel_v22/articulated/REMOTE_CONFIG_MANIFEST.json"
    manifest.write_text(json.dumps({"schema":"graph-self-adapter.articulated-remote-config-manifest.v1","records":records,"key_content_read":False,"formal_episode_started":False},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"manifest":str(manifest),"sha256":sha(manifest),"records":records},indent=2))

if __name__=="__main__":main()
