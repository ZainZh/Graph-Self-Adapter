"""B2 constant-safe P59 readout pretraining and meta-training CLI.

This command performs no simulator or robot rollout.  ``prepare`` seals the
dataset and implementation; ``train`` executes the single seed11 final-only
run after root review of the freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .contracts import P59Contract, PublicTransition
from .core import geometry_loss
from .core_batched import BatchedP59AdaptationCore as P59AdaptationCore
from .history import HistoryCorrection, history_outer_loss


ROOT = Path(__file__).resolve().parents[3]
MODEL_CONTRACT = ROOT / "evaluations/fast_adapter_p59_control_latent_v01/MODEL_CONTRACT_B2.json"
P50_CHECKPOINT = ROOT / "evaluations/fast_adapter_p50_precision_policy_v01/offline_v01/precision_action.pt"
SOURCES = [
    "prove/task_graph/p59/contracts.py",
    "prove/task_graph/p59/core.py",
    "prove/task_graph/p59/core_batched.py",
    "prove/task_graph/p59/train_constant_safe.py",
    "prove/task_graph/p59/history.py",
    "prove/task_graph/p59/checkpoint.py",
    "prove/fast_adapter/p59_data.py",
    "tools/fast_adapter/p59_materialize.py",
    "tests/test_p59_adaptation_core.py",
    "tests/test_p59_constant_safe.py",
    "evaluations/fast_adapter_p59_control_latent_v01/MODEL_CONTRACT_B2.json",
    "evaluations/fast_adapter_p59_control_latent_v01/AMENDMENT_B2_CONSTANT_NORMALIZATION.md",
    "evaluations/fast_adapter_p59_control_latent_v01/EXECUTION_CONTRACT_BATCHED.json",
    "evaluations/fast_adapter_p59_control_latent_v01/speed_check_v01/BATCHED_BACKEND_BENCHMARK.json",
]

CONSTANT_STD_THRESHOLD = 1e-4


def constant_safe_statistics(all_train_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return train-only mean, B2 scale, and constant-column mask."""
    if all_train_u.ndim != 2 or not torch.isfinite(all_train_u).all():
        raise ValueError("normalization frames must be a finite matrix")
    mean = all_train_u.mean(0)
    std = all_train_u.std(0, unbiased=False)
    constant = std <= CONSTANT_STD_THRESHOLD
    scale = torch.where(constant, torch.ones_like(std), std)
    return mean, scale, constant


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def append_log(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def load_contract() -> dict:
    value = json.loads(MODEL_CONTRACT.read_text())
    if value.get("schema") != "prove.p59-model-contract.v1":
        raise ValueError("unknown P59 model contract")
    return value


def load_dataset(path: Path, expected_u_dim: int) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != "prove.p59-training-dataset.v1":
        raise ValueError("unknown P59 dataset schema")
    if value.get("u_dim") != expected_u_dim or value.get("feature_variant") != "fusion_mask196_dino_pca48":
        raise ValueError("dataset public representation differs from MODEL_CONTRACT")
    if set(value.get("splits", {})) != {"train", "developer"}:
        raise ValueError("dataset must keep train and opened developer splits separate")
    required = {"episode_id", "readout_u", "readout_oracle_geometry", "u", "q",
                "executed_world_delta6", "oracle_geometry",
                "oracle_p50_action6", "step_index"}
    provenance = value.get("feature_provenance", {})
    for key in ("dataset_manifest_sha256", "relative_fusion_pca_sha256"):
        if len(provenance.get(key, "")) != 64:
            raise ValueError(f"missing bound feature provenance: {key}")
    expected_counts = {"train": 96, "developer": 12}
    for split, episodes in value["splits"].items():
        if len(episodes) != expected_counts[split]:
            raise ValueError(f"{split} episode count differs from frozen contract")
        identities = set()
        for episode in episodes:
            if not required.issubset(episode) or episode["episode_id"] in identities:
                raise ValueError("episode fields or identity invalid")
            identities.add(episode["episode_id"])
            shapes = {
                "readout_u": (64, expected_u_dim), "readout_oracle_geometry": (64, 12),
                "u": (25, expected_u_dim), "q": (25, 9),
                "executed_world_delta6": (24, 6),
                "oracle_geometry": (25, 12), "oracle_p50_action6": (25, 4, 6),
                "step_index": (25,),
            }
            for name, shape in shapes.items():
                tensor = episode[name]
                if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                    raise ValueError(f"{split}/{episode['episode_id']}/{name} must be finite {shape}")
            steps = episode["step_index"].to(torch.long)
            if not torch.equal(steps, torch.arange(25)):
                raise ValueError("only verified continuous t0..t24 may enter P59 training")
    return value


def prepare(dataset_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "TRAINING_FREEZE.json"
    if freeze_path.exists():
        raise FileExistsError("P59 training already prepared")
    model_contract = load_contract()
    load_dataset(dataset_path, model_contract["public_representation"]["dimension"])
    freeze = {
        "schema": "prove.p59-training-freeze.v1",
        "model_contract": model_contract,
        "model_contract_sha256": sha(MODEL_CONTRACT),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha(dataset_path),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "p50_checkpoint": str(P50_CHECKPOINT.relative_to(ROOT)),
        "p50_checkpoint_sha256": sha(P50_CHECKPOINT),
        "training_started": False,
        "physics_rollouts": 0,
    }
    dump_new(freeze_path, freeze)
    return freeze


def verify_freeze(dataset_path: Path, output: Path) -> tuple[dict, dict]:
    freeze = json.loads((output / "TRAINING_FREEZE.json").read_text())
    if freeze.get("schema") != "prove.p59-training-freeze.v1":
        raise ValueError("unknown training freeze")
    if sha(dataset_path) != freeze["dataset_sha256"] or sha(MODEL_CONTRACT) != freeze["model_contract_sha256"]:
        raise ValueError("dataset or model contract changed after freeze")
    for name, digest in freeze["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"implementation changed after freeze: {name}")
    if sha(P50_CHECKPOINT) != freeze["p50_checkpoint_sha256"]:
        raise ValueError("P50 checkpoint changed after freeze")
    dataset = load_dataset(dataset_path, freeze["model_contract"]["public_representation"]["dimension"])
    return freeze, dataset


def _stack_frames(episodes: list[dict], episode_indices: torch.Tensor,
                  frame_indices: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    return torch.stack([episodes[int(e)][name][int(t)] for e, t in zip(episode_indices, frame_indices)]).to(device)


def _batched_support(episodes: list[dict], episode_indices: torch.Tensor,
                     query_step: int, device: torch.device) -> list[PublicTransition]:
    batch = len(episode_indices)
    result = []
    for step in range(query_step):
        u = torch.stack([episodes[int(e)]["u"][step] for e in episode_indices]).to(device)
        q = torch.stack([episodes[int(e)]["q"][step] for e in episode_indices]).to(device)
        action = torch.stack([episodes[int(e)]["executed_world_delta6"][step] for e in episode_indices]).to(device)
        next_u = torch.stack([episodes[int(e)]["u"][step + 1] for e in episode_indices]).to(device)
        next_q = torch.stack([episodes[int(e)]["q"][step + 1] for e in episode_indices]).to(device)
        # Support is public pre-contact observation motion, not a physical P50
        # phase. Relation feedback excludes phase, so a fixed valid index is
        # used only to satisfy the shared P50 feature constructor.
        phase = torch.zeros(batch, dtype=torch.long, device=device)
        next_phase = torch.zeros(batch, dtype=torch.long, device=device)
        result.append(PublicTransition(u, q, action, next_u, next_q, phase, next_phase,
                                       step, step + 1, f"batch-of-{batch}-aligned-episodes"))
    return result


def train(dataset_path: Path, output: Path, device_name: str) -> dict:
    freeze, dataset = verify_freeze(dataset_path, output)
    contract = freeze["model_contract"]
    if device_name != contract["training"]["device"]:
        raise ValueError("device differs from MODEL_CONTRACT")
    if any((output / name).exists() for name in ("TRAINING_START.json", "TRAINING_LOG.jsonl", "public_readout.pt", "p59_model.pt", "TRAINING_SEAL.json", "ABORT.json")):
        raise FileExistsError("P59 training already attempted; no overwrite/retry")
    dump_new(output / "TRAINING_START.json", {
        "schema": "prove.p59-training-start.v1", "freeze_sha256": sha(output / "TRAINING_FREEZE.json"),
        "seed": contract["training"]["seed"], "automatic_retries": 0,
    })
    (output / "TRAINING_LOG.jsonl").touch(exist_ok=False)
    log_path = output / "TRAINING_LOG.jsonl"
    device = torch.device(device_name)
    seed = contract["training"]["seed"]
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    episodes = dataset["splits"]["train"]
    p59_contract = P59Contract(u_dim=contract["public_representation"]["dimension"])
    model = P59AdaptationCore(p59_contract, P50_CHECKPOINT).to(device)
    all_train_u = torch.cat([episode["readout_u"] for episode in episodes], 0)
    u_mean, u_scale, constant_columns = constant_safe_statistics(all_train_u)
    u_mean, u_scale = u_mean.to(device), u_scale.to(device)
    model.set_public_normalization(u_mean, u_scale)
    started = time.perf_counter()

    readout_cfg = contract["training"]["readout"]
    optimizer = torch.optim.Adam(model.public_readout.parameters(), lr=readout_cfg["learning_rate"])
    model.train()
    for update in range(readout_cfg["updates"]):
        episode_indices = torch.randint(len(episodes), (readout_cfg["batch_size"],), generator=generator)
        frame_indices = torch.randint(64, (readout_cfg["batch_size"],), generator=generator)
        u = _stack_frames(episodes, episode_indices, frame_indices, "readout_u", device)
        target = _stack_frames(episodes, episode_indices, frame_indices, "readout_oracle_geometry", device)
        optimizer.zero_grad(set_to_none=True)
        readout_loss = geometry_loss(model.public_geometry(u), target)
        if not torch.isfinite(readout_loss):
            raise FloatingPointError(f"nonfinite readout loss at update {update + 1}")
        readout_loss.backward()
        optimizer.step()
        if (update + 1) % 100 == 0:
            append_log(log_path, {"stage": "readout", "update": update + 1,
                                  "loss": float(readout_loss.detach()), "finite": True})
    with torch.no_grad():
        readout_sse = 0.
        readout_examples = 0
        for start in range(0, len(all_train_u), 512):
            u_batch = all_train_u[start:start + 512].to(device)
            target_batch = torch.cat([episode["readout_oracle_geometry"] for episode in episodes], 0)[start:start + 512].to(device)
            count = len(u_batch)
            readout_sse += float(geometry_loss(model.public_geometry(u_batch), target_batch)) * count
            readout_examples += count
        readout_train_scaled_geometry_loss = readout_sse / readout_examples
    append_log(log_path, {"stage": "readout_train_summary", "frames": readout_examples,
                          "scaled_geometry_loss": readout_train_scaled_geometry_loss,
                          "developer_frames_read": 0, "finite": True})
    readout_payload = {
        "schema": "prove.p59-frozen-public-readout.v1",
        "state_dict": model.public_readout.state_dict(),
        "model_contract_sha256": sha(MODEL_CONTRACT),
        "updates": readout_cfg["updates"],
        "u_mean": model.u_mean.detach().cpu(),
        "u_scale": model.u_scale.detach().cpu(),
        "normalization_source_frames": len(episodes) * 64,
        "normalization_rule": "scale=1 if train_std<=1e-4 else train_std",
        "normalization_constant_std_threshold": CONSTANT_STD_THRESHOLD,
        "normalization_constant_columns": int(constant_columns.sum()),
        "train_scaled_geometry_loss": readout_train_scaled_geometry_loss,
    }
    with (output / "public_readout.pt").open("xb") as stream:
        torch.save(readout_payload, stream)
    model.freeze_public_readout()

    meta_cfg = contract["training"]["meta"]
    parameters = list(model.initializer.parameters()) + list(model.feedback_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=meta_cfg["learning_rate"])
    for update in range(meta_cfg["updates"]):
        episode_indices = torch.randint(len(episodes), (meta_cfg["batch_size"],), generator=generator)
        query_step = int(torch.randint(1, 25, (), generator=generator))
        support = _batched_support(episodes, episode_indices, query_step, device)
        query_indices = torch.full((meta_cfg["batch_size"],), query_step)
        query_u = _stack_frames(episodes, episode_indices, query_indices, "u", device)
        query_q = _stack_frames(episodes, episode_indices, query_indices, "q", device)
        query_phase = torch.arange(meta_cfg["batch_size"], device=device) % 4
        all_oracle_actions = _stack_frames(episodes, episode_indices, query_indices, "oracle_p50_action6", device)
        oracle_action = all_oracle_actions[torch.arange(meta_cfg["batch_size"], device=device), query_phase]
        oracle_geometry = _stack_frames(episodes, episode_indices, query_indices, "oracle_geometry", device)
        optimizer.zero_grad(set_to_none=True)
        meta_loss, parts = model.outer_loss(
            support, query_u, query_q, query_phase, oracle_action, oracle_geometry,
            query_step_index=query_step, query_episode_id=f"batch-of-{meta_cfg['batch_size']}-aligned-episodes")
        if not torch.isfinite(meta_loss):
            raise FloatingPointError(f"nonfinite meta loss at update {update + 1}")
        meta_loss.backward()
        optimizer.step()
        if (update + 1) % 100 == 0:
            append_log(log_path, {"stage": "meta", "update": update + 1,
                                  "loss": float(meta_loss.detach()),
                                  "online_world_bc": float(parts["online_bc"].detach()),
                                  "fixed_world_bc": float(parts["fixed_bc"].detach()), "finite": True})

    history_model = HistoryCorrection(history=p59_contract.history).to(device)
    optimizer = torch.optim.Adam(history_model.parameters(), lr=meta_cfg["learning_rate"])
    for update in range(meta_cfg["updates"]):
        episode_indices = torch.randint(len(episodes), (meta_cfg["batch_size"],), generator=generator)
        query_step = int(torch.randint(1, 25, (), generator=generator))
        support = _batched_support(episodes, episode_indices, query_step, device)
        query_indices = torch.full((meta_cfg["batch_size"],), query_step)
        query_u = _stack_frames(episodes, episode_indices, query_indices, "u", device)
        query_q = _stack_frames(episodes, episode_indices, query_indices, "q", device)
        query_phase = torch.arange(meta_cfg["batch_size"], device=device) % 4
        all_oracle_actions = _stack_frames(episodes, episode_indices, query_indices, "oracle_p50_action6", device)
        oracle_action = all_oracle_actions[torch.arange(meta_cfg["batch_size"], device=device), query_phase]
        oracle_geometry = _stack_frames(episodes, episode_indices, query_indices, "oracle_geometry", device)
        optimizer.zero_grad(set_to_none=True)
        history_loss, history_parts = history_outer_loss(
            model, history_model, support, query_u, query_q, query_phase,
            oracle_action, oracle_geometry, query_step_index=query_step,
            query_episode_id=f"batch-of-{meta_cfg['batch_size']}-aligned-episodes")
        if not torch.isfinite(history_loss):
            raise FloatingPointError(f"nonfinite History loss at update {update + 1}")
        history_loss.backward()
        optimizer.step()
        if (update + 1) % 100 == 0:
            append_log(log_path, {"stage": "history", "update": update + 1,
                                  "loss": float(history_loss.detach()),
                                  "world_bc": float(history_parts["world_action"].detach()), "finite": True})

    model.freeze_runtime()
    payload = {
        "schema": "prove.p59-control-latent-checkpoint.v1",
        "state_dict": model.state_dict(), "contract": contract,
        "history_state_dict": history_model.state_dict(),
        "training_freeze_sha256": sha(output / "TRAINING_FREEZE.json"),
    }
    with (output / "p59_model.pt").open("xb") as stream:
        torch.save(payload, stream)
    seal = {
        "schema": "prove.p59-training-seal.v1",
        "public_readout_sha256": sha(output / "public_readout.pt"),
        "checkpoint_sha256": sha(output / "p59_model.pt"),
        "readout_updates": readout_cfg["updates"], "meta_updates": meta_cfg["updates"],
        "final_readout_loss": float(readout_loss.detach()), "final_meta_loss": float(meta_loss.detach()),
        "readout_train_scaled_geometry_loss": readout_train_scaled_geometry_loss,
        "final_online_world_bc": float(parts["online_bc"].detach()),
        "final_fixed_world_bc": float(parts["fixed_bc"].detach()),
        "history_updates": meta_cfg["updates"],
        "final_history_loss": float(history_loss.detach()),
        "final_history_world_bc": float(history_parts["world_action"].detach()),
        "elapsed_seconds": time.perf_counter() - started, "physics_rollouts": 0,
        "consumed": {
            "feature_variant": dataset["feature_variant"],
            "u_dim": dataset["u_dim"],
            "train_readout_frames": 6144,
            "train_continuous_transitions": 2304,
            "developer_frames_for_selection": 0,
            "phase_balanced_queries_per_meta_batch": [4, 4, 4, 4],
            "transition_scope": "precontact_observation_motion_only",
        },
        "developer_split_opened_for_selection": False,
        "normalization_rule": "scale=1 if train_std<=1e-4 else train_std",
        "normalization_constant_std_threshold": CONSTANT_STD_THRESHOLD,
        "normalization_constant_columns": int(constant_columns.sum()),
    }
    dump_new(output / "TRAINING_SEAL.json", seal)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    try:
        result = prepare(args.dataset, args.output) if args.mode == "prepare" else train(args.dataset, args.output, args.device)
    except Exception as error:
        abort = args.output / "ABORT.json"
        if args.mode == "train" and args.output.exists() and not abort.exists():
            dump_new(abort, {"schema": "prove.p59-training-abort.v1",
                             "error_type": type(error).__name__, "message": str(error),
                             "automatic_retry": False, "physics_rollouts": 0})
        raise
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
