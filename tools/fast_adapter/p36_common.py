"""P36 RGB-geometry token ablation paths and source bindings."""
from pathlib import Path
import json
import torch

from tools.fast_adapter.p33_common import dump, digest, bindings, base_model, base_path, state_hash, DINO

ROOT = Path("evaluations/fast_adapter_p36_geometry_tokens_v01")
TASKS = ("mug-handle-lift", "hammer-block")
MODES = ("no_explicit_bbox_geometry", "p34_existing_bbox_geometry", "p36_explicit_relation_geometry")
H = 4


def protocol():
    return json.loads((ROOT / "PROTOCOL.json").read_text())


def bank_path(task):
    if task == "mug-handle-lift":
        return Path("evaluations/fast_adapter_p34_role_tokens_v01/mug-handle-lift/data/BANK.pt")
    return Path("evaluations/fast_adapter_p33_relation_dynamics_v01/data/BANK.pt")


def load_data(task, device="cuda"):
    bank = torch.load(bank_path(task), weights_only=True)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in bank.items()}


def stats_path(task):
    return ROOT / task / "data/STATS.pt"


def source_paths(task):
    return [
        ROOT / "PROTOCOL.json",
        base_path(task),
        bank_path(task),
        stats_path(task),
        Path("prove/fast_adapter/geometry_role_dynamics.py"),
        Path("prove/fast_adapter/role_dynamics.py"),
        Path("prove/fast_adapter/relational_dynamics.py"),
        Path("tools/fast_adapter/p36_common.py"),
        Path("tools/fast_adapter/p36_train.py"),
        Path("tools/fast_adapter/p34_visual.py"),
        Path("tools/fast_adapter/p33_visual.py"),
        Path("tools/fast_adapter/p29_common.py"),
        Path("prove/fast_adapter/lila_frontend.py"),
        Path("prove/fast_adapter/object_interaction_env.py"),
        Path("prove/fast_adapter/held_hammer_env.py"),
    ]


def model(task, seed, mode, device="cuda"):
    from prove.fast_adapter.geometry_role_dynamics import GeometryRoleDynamics

    net = GeometryRoleDynamics(mode, task).to(device)
    path = ROOT / task / "models" / f"seed{seed}" / f"{mode}.pt"
    net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["state"])
    return net.eval().requires_grad_(False)
