"""Prepare and train the preregistered P59 FullHistory sidecar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .checkpoint import load_frozen_checkpoint
from .contracts import PublicTransition
from .full_history import FullHistoryReference, full_history_outer_loss
from .train import load_dataset


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "evaluations/fast_adapter_p59_control_latent_v01/FULL_HISTORY_CONTRACT.json"
SOURCES = (
    "prove/task_graph/p59/contracts.py",
    "prove/task_graph/p59/core.py",
    "prove/task_graph/p59/checkpoint.py",
    "prove/task_graph/p59/train.py",
    "prove/task_graph/p59/full_history.py",
    "prove/task_graph/p59/full_history_train.py",
    "tests/test_p59_full_history.py",
    "evaluations/fast_adapter_p59_control_latent_v01/FULL_HISTORY_CONTRACT.json",
)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def append_log(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def load_contract() -> dict:
    value = json.loads(CONTRACT.read_text())
    if value.get("schema") != "prove.p59-full-history-contract.v1":
        raise ValueError("unknown FullHistory contract")
    return value


def _verify_main_seal(checkpoint: Path, seal_path: Path) -> dict:
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != "prove.p59-training-seal.v1":
        raise ValueError("unknown main P59 training seal")
    if seal.get("checkpoint_sha256") != sha(checkpoint):
        raise ValueError("main P59 checkpoint differs from its seal")
    if seal.get("developer_split_opened_for_selection") is not False:
        raise ValueError("FullHistory cannot use a developer-selected main checkpoint")
    return seal


def prepare(dataset_path: Path, main_checkpoint: Path, main_seal: Path,
            p50_checkpoint: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "FULL_HISTORY_FREEZE.json"
    if freeze_path.exists():
        raise FileExistsError("FullHistory is already prepared")
    contract = load_contract()
    dataset = load_dataset(dataset_path, 244)
    _verify_main_seal(main_checkpoint, main_seal)
    if len(dataset["splits"]["train"]) != contract["training"]["episodes"]:
        raise ValueError("FullHistory training episode count differs from contract")
    freeze = {
        "schema": "prove.p59-full-history-freeze.v1",
        "contract_sha256": sha(CONTRACT),
        "dataset_path": str(dataset_path), "dataset_sha256": sha(dataset_path),
        "main_checkpoint_path": str(main_checkpoint),
        "main_checkpoint_sha256": sha(main_checkpoint),
        "main_training_seal_path": str(main_seal),
        "main_training_seal_sha256": sha(main_seal),
        "p50_checkpoint_path": str(p50_checkpoint),
        "p50_checkpoint_sha256": sha(p50_checkpoint),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "training_started": False, "developer_frames_read": 0, "physics_rollouts": 0,
    }
    dump_new(freeze_path, freeze)
    return freeze


def verify_freeze(dataset_path: Path, main_checkpoint: Path, main_seal: Path,
                  p50_checkpoint: Path, output: Path) -> tuple[dict, dict, dict]:
    freeze = json.loads((output / "FULL_HISTORY_FREEZE.json").read_text())
    if freeze.get("schema") != "prove.p59-full-history-freeze.v1":
        raise ValueError("unknown FullHistory freeze")
    actual = {
        "dataset_sha256": sha(dataset_path), "main_checkpoint_sha256": sha(main_checkpoint),
        "main_training_seal_sha256": sha(main_seal),
        "p50_checkpoint_sha256": sha(p50_checkpoint), "contract_sha256": sha(CONTRACT),
    }
    for key, digest in actual.items():
        if freeze.get(key) != digest:
            raise ValueError(f"FullHistory frozen input changed: {key}")
    for name, digest in freeze["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"FullHistory implementation changed after freeze: {name}")
    seal = _verify_main_seal(main_checkpoint, main_seal)
    dataset = load_dataset(dataset_path, 244)
    return freeze, seal, dataset


def _frames(episodes: list[dict], episode_indices: torch.Tensor,
            frame_index: int, name: str, device) -> torch.Tensor:
    return torch.stack([episodes[int(index)][name][frame_index]
                        for index in episode_indices]).to(device)


def _support(episodes: list[dict], episode_indices: torch.Tensor,
             query_step: int, device) -> list[PublicTransition]:
    batch = len(episode_indices)
    phase = torch.zeros(batch, dtype=torch.long, device=device)
    result = []
    for step in range(query_step):
        result.append(PublicTransition(
            _frames(episodes, episode_indices, step, "u", device),
            _frames(episodes, episode_indices, step, "q", device),
            _frames(episodes, episode_indices, step, "executed_world_delta6", device),
            _frames(episodes, episode_indices, step + 1, "u", device),
            _frames(episodes, episode_indices, step + 1, "q", device),
            phase, phase, step, step + 1, f"full-history-batch-{batch}"))
    return result


def train(dataset_path: Path, main_checkpoint: Path, main_seal: Path,
          p50_checkpoint: Path, output: Path, device_name: str) -> dict:
    freeze, _, dataset = verify_freeze(
        dataset_path, main_checkpoint, main_seal, p50_checkpoint, output)
    contract = load_contract()
    if any((output / name).exists() for name in
           ("FULL_HISTORY_START.json", "FULL_HISTORY_LOG.jsonl",
            "full_history.pt", "FULL_HISTORY_SEAL.json", "FULL_HISTORY_ABORT.json")):
        raise FileExistsError("FullHistory training already attempted")
    dump_new(output / "FULL_HISTORY_START.json", {
        "schema": "prove.p59-full-history-start.v1",
        "freeze_sha256": sha(output / "FULL_HISTORY_FREEZE.json"),
        "seed": contract["training"]["seed"], "automatic_retries": 0,
    })
    log_path = output / "FULL_HISTORY_LOG.jsonl"
    log_path.touch(exist_ok=False)
    device = torch.device(device_name)
    seed = contract["training"]["seed"]
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    core, _, main_contract = load_frozen_checkpoint(
        main_checkpoint, p50_checkpoint, device=device_name)
    if main_contract["public_representation"]["dimension"] != 244:
        raise ValueError("FullHistory requires the frozen fusion U244 main candidate")
    reference = FullHistoryReference(244, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(reference.parameters(), lr=contract["training"]["learning_rate"])
    episodes = dataset["splits"]["train"]
    batch_size = contract["training"]["batch_size"]
    started = time.perf_counter()
    reference.train()
    for update in range(contract["training"]["updates"]):
        episode_indices = torch.randint(len(episodes), (batch_size,), generator=generator)
        query_step = int(torch.randint(1, 25, (), generator=generator))
        support = _support(episodes, episode_indices, query_step, device)
        first_u = _frames(episodes, episode_indices, 0, "u", device)
        query_u = _frames(episodes, episode_indices, query_step, "u", device)
        query_q = _frames(episodes, episode_indices, query_step, "q", device)
        query_geometry = _frames(episodes, episode_indices, query_step, "oracle_geometry", device)
        all_actions = _frames(episodes, episode_indices, query_step, "oracle_p50_action6", device)
        query_phase = torch.arange(batch_size, device=device) % 4
        query_action = all_actions[torch.arange(batch_size, device=device), query_phase]
        optimizer.zero_grad(set_to_none=True)
        loss, parts = full_history_outer_loss(
            core, reference, first_u, support, query_u, query_q, query_phase,
            query_action, query_geometry, geometry_weight=.25)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite FullHistory loss at update {update + 1}")
        loss.backward()
        optimizer.step()
        if (update + 1) % 100 == 0:
            append_log(log_path, {
                "update": update + 1, "loss": float(loss.detach()),
                "world_action": float(parts["world_action"].detach()),
                "geometry": float(parts["geometry"].detach()),
                "developer_frames_read": 0, "finite": True,
            })
    reference.eval().requires_grad_(False)
    payload = {
        "schema": "prove.p59-full-history-checkpoint.v1",
        "state_dict": reference.state_dict(), "u_dim": 244, "hidden_dim": 128,
        "contract_sha256": sha(CONTRACT),
        "main_checkpoint_sha256": sha(main_checkpoint),
        "full_history_freeze_sha256": sha(output / "FULL_HISTORY_FREEZE.json"),
    }
    with (output / "full_history.pt").open("xb") as stream:
        torch.save(payload, stream)
    seal = {
        "schema": "prove.p59-full-history-seal.v1",
        "checkpoint_sha256": sha(output / "full_history.pt"),
        "updates": contract["training"]["updates"], "seed": seed,
        "final_loss": float(loss.detach()),
        "final_world_action": float(parts["world_action"].detach()),
        "final_geometry": float(parts["geometry"].detach()),
        "elapsed_seconds": time.perf_counter() - started,
        "train_episodes": len(episodes), "developer_frames_read_for_selection": 0,
        "physics_rollouts": 0, "automatic_retries": 0,
    }
    dump_new(output / "FULL_HISTORY_SEAL.json", seal)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--main-checkpoint", type=Path, required=True)
    parser.add_argument("--main-seal", type=Path, required=True)
    parser.add_argument("--p50-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.dataset, args.main_checkpoint, args.main_seal,
                args.p50_checkpoint, args.output)
    else:
        train(args.dataset, args.main_checkpoint, args.main_seal,
              args.p50_checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
