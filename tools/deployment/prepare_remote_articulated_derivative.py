#!/usr/bin/env python3
"""Prepare the isolated ubuntu server derivative; never run on source workspace."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket


GRAPH = Path("/home/ubuntu/github/Graph-Self-Adapter-v22")
MUJOCO = Path("/home/ubuntu/github/Mujoco-RSI-v22")
PYTHON = "/home/ubuntu/venvs/graph-v22/bin/python"
MAPPINGS = (
    ("/home/zheng/github/xtrainer_clover/dobot_control/robots/dobot_api.py", "/home/ubuntu/github/RoboTwin_bin_pick/include/xtrainer_clover/dobot_control/robots/dobot_api.py"),
    ("/home/zheng/github/Graph-Self-Adapter", str(GRAPH)),
    ("/home/zheng/github/Mujoco-RSI", str(MUJOCO)),
    ("/home/zheng/github/RoboTwin_bin_pick", "/home/ubuntu/github/RoboTwin_bin_pick"),
    ("/home/zheng/github/SAM2_streaming", "/home/ubuntu/github/SAM2_streaming"),
    ("/home/zheng/Datasets/sam2/sam2.1_hiera_small.pt", "/home/ubuntu/Datasets/sam2/sam2.1_hiera_small.pt"),
    ("/home/zheng/.config/graph-self-adapter/openrouter.key", "/home/ubuntu/.config/graph-self-adapter/openrouter.key"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rewrite_text(value):
    for source, destination in MAPPINGS:
        value = value.replace(source, destination)
    return value


def rewrite_json(value):
    if isinstance(value, str):
        return rewrite_text(value)
    if isinstance(value, list):
        return [rewrite_json(item) for item in value]
    if isinstance(value, dict):
        return {rewrite_text(str(key)): rewrite_json(item) for key, item in value.items()}
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    if socket.gethostname() == socket.gethostbyname(socket.gethostname()) or not GRAPH.is_dir() or not MUJOCO.is_dir():
        # Path identity is the real guard; hostname layouts vary on the server.
        if not str(Path(__file__).resolve()).startswith(str(GRAPH) + "/"):
            raise SystemExit("remote derivative preparer must run from the isolated ubuntu graph root")
    records = []

    # Rewrite every available task scene in the isolated Mujoco derivative.
    scene_hashes = {}
    for scene in sorted((MUJOCO / "runs/physics_v2").glob("*/scene.xml")):
        before = sha(scene)
        text = rewrite_text(scene.read_text())
        scene.write_text(text)
        after = sha(scene)
        scene_hashes[str(scene)] = after
        records.append({"path": str(scene), "source_sha256": before, "derivative_sha256": after, "kind": "scene_xml_path_rewrite"})

    inventory = GRAPH / "evaluations/multitask_runtime_v20/TASK_ASSET_INVENTORY.json"
    before = sha(inventory)
    write_json(inventory, rewrite_json(json.loads(inventory.read_text())))
    records.append({"path": str(inventory), "source_sha256": before, "derivative_sha256": sha(inventory), "kind": "inventory_path_rewrite"})

    catalog = GRAPH / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json"
    before = sha(catalog)
    value = rewrite_json(json.loads(catalog.read_text()))
    for row in value["task_assets"].values():
        hashes = row["external_asset_hashes"]
        for path in list(hashes):
            if path in scene_hashes:
                hashes[path] = scene_hashes[path]
    write_json(catalog, value)
    records.append({"path": str(catalog), "source_sha256": before, "derivative_sha256": sha(catalog), "kind": "catalog_path_and_scene_hash_rewrite"})

    tasks = ("hanging_mug", "open_drawer_and_place_object")
    configs = {}
    for task in tasks:
        source = GRAPH / "evaluations/multitask_runtime_v20" / task / "CONFIG_PROPOSED.json"
        target = GRAPH / "evaluations/multitask_parallel_v22/articulated" / task / "server_fair_v22_3/CONFIG_SERVER.json"
        cfg = rewrite_json(json.loads(source.read_text()))
        write_json(target, cfg)
        configs[task] = {"path": str(target), "sha256": sha(target), "source_path": str(source), "source_sha256": sha(source)}
        records.append({"path": str(target), "source_sha256": sha(source), "derivative_sha256": sha(target), "kind": "server_config_derivative"})

    for task in tasks:
        launcher = GRAPH / "task_packages/multitask_parallel_v22" / task / "run_server_fair_v22_3.sh"
        before = sha(launcher)
        text = launcher.read_text().replace("/home/zheng/miniforge3/envs/geometry/bin/python", PYTHON)
        launcher.write_text(text)
        launcher.chmod(0o755)
        records.append({"path": str(launcher), "source_sha256": before, "derivative_sha256": sha(launcher), "kind": "server_python_binding"})

    manifest = {
        "schema": "graph-self-adapter.remote-rewrite-manifest.v1",
        "status": "derivative_paths_rewritten_not_executed",
        "source_host_paths_preserved_elsewhere": True,
        "remote_graph_root": str(GRAPH),
        "remote_mujoco_root": str(MUJOCO),
        "python": PYTHON,
        "mappings": [{"source": source, "destination": destination} for source, destination in MAPPINGS],
        "records": records,
        "configs": configs,
        "formal_episode_started": False,
    }
    target = GRAPH / "evaluations/multitask_parallel_v22/REMOTE_REWRITE_MANIFEST.json"
    write_json(target, manifest)
    print(json.dumps({"manifest": str(target), "sha256": sha(target), "records": len(records), "configs": configs}, indent=2))


if __name__ == "__main__":
    main()
