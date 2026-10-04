"""Offline sidecar evaluation for the preregistered P59 FullHistory reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .contracts import PublicTransition
from .evaluate import (
    FUTURE_STEPS,
    _aggregate,
    _query_record,
    _to,
    _validate_evaluation_fields,
    load_sealed_candidate,
)
from .full_history import load_full_history_checkpoint
from .history import cyclic_wrong_history_map
from .train import load_dataset


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "evaluations/fast_adapter_p59_control_latent_v01/FULL_HISTORY_CONTRACT.json"
EVAL_SOURCES = (
    "prove/task_graph/p59/full_history_evaluate.py",
    "prove/task_graph/p59/evaluate.py",
    "prove/task_graph/p59/full_history.py",
    "tests/test_p59_full_history.py",
    "evaluations/fast_adapter_p59_control_latent_v01/FULL_HISTORY_CONTRACT.json",
)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _verify_sidecar(sidecar_checkpoint: Path, sidecar_seal: Path,
                    sidecar_training_freeze: Path, main_checkpoint: Path,
                    dataset_path: Path) -> dict:
    seal = json.loads(sidecar_seal.read_text())
    freeze = json.loads(sidecar_training_freeze.read_text())
    if seal.get("schema") != "prove.p59-full-history-seal.v1":
        raise ValueError("unknown FullHistory training seal")
    if freeze.get("schema") != "prove.p59-full-history-freeze.v1":
        raise ValueError("unknown FullHistory training freeze")
    if seal.get("checkpoint_sha256") != sha(sidecar_checkpoint):
        raise ValueError("FullHistory checkpoint differs from its training seal")
    if seal.get("developer_frames_read_for_selection") != 0:
        raise ValueError("developer-selected FullHistory checkpoints are forbidden")
    if freeze.get("dataset_sha256") != sha(dataset_path):
        raise ValueError("FullHistory evaluation dataset differs from training")
    if freeze.get("main_checkpoint_sha256") != sha(main_checkpoint):
        raise ValueError("FullHistory training used a different main candidate")
    return {"sidecar_checkpoint_sha256": sha(sidecar_checkpoint),
            "sidecar_training_freeze_sha256": sha(sidecar_training_freeze)}


def prepare_evaluation(dataset_path: Path, main_checkpoint: Path, main_seal: Path,
                       main_training_freeze: Path, p50_checkpoint: Path,
                       sidecar_checkpoint: Path, sidecar_seal: Path,
                       sidecar_training_freeze: Path, output: Path) -> dict:
    """Freeze all model/data/source identities before opening evaluation."""
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "FULL_HISTORY_EVAL_FREEZE.json"
    if freeze_path.exists():
        raise FileExistsError("FullHistory evaluation is already prepared")
    core, _, main_provenance = load_sealed_candidate(
        main_checkpoint, main_seal, main_training_freeze, p50_checkpoint, device="cpu")
    dataset = load_dataset(dataset_path, core.contract.u_dim)
    _validate_evaluation_fields(dataset)
    if sha(dataset_path) != main_provenance["training_dataset_sha256"]:
        raise ValueError("main candidate was trained on a different dataset")
    sidecar_provenance = _verify_sidecar(
        sidecar_checkpoint, sidecar_seal, sidecar_training_freeze,
        main_checkpoint, dataset_path)
    load_full_history_checkpoint(sidecar_checkpoint, main_checkpoint, device="cpu")
    value = {
        "schema": "prove.p59-full-history-evaluation-freeze.v1",
        "dataset_sha256": sha(dataset_path),
        "main_checkpoint_sha256": sha(main_checkpoint),
        "main_training_seal_sha256": sha(main_seal),
        "main_training_freeze_sha256": sha(main_training_freeze),
        "p50_checkpoint_sha256": sha(p50_checkpoint),
        "sidecar_checkpoint_sha256": sidecar_provenance["sidecar_checkpoint_sha256"],
        "sidecar_training_seal_sha256": sha(sidecar_seal),
        "sidecar_training_freeze_sha256": sidecar_provenance["sidecar_training_freeze_sha256"],
        "contract_sha256": sha(CONTRACT),
        "source_sha256": {name: sha(ROOT / name) for name in EVAL_SOURCES},
        "methods": ["full_history", "wrong_full_history"],
        "wrong_context": "recipient U0 initialization plus fixed cyclic donor transitions from the first arrived step",
        "developer_used_for_selection": False,
        "physics_rollouts": 0,
    }
    dump_new(freeze_path, value)
    return value


def verify_evaluation_freeze(dataset_path: Path, main_checkpoint: Path, main_seal: Path,
                             main_training_freeze: Path, p50_checkpoint: Path,
                             sidecar_checkpoint: Path, sidecar_seal: Path,
                             sidecar_training_freeze: Path, output: Path) -> dict:
    value = json.loads((output / "FULL_HISTORY_EVAL_FREEZE.json").read_text())
    if value.get("schema") != "prove.p59-full-history-evaluation-freeze.v1":
        raise ValueError("unknown FullHistory evaluation freeze")
    expected = {
        "dataset_sha256": sha(dataset_path),
        "main_checkpoint_sha256": sha(main_checkpoint),
        "main_training_seal_sha256": sha(main_seal),
        "main_training_freeze_sha256": sha(main_training_freeze),
        "p50_checkpoint_sha256": sha(p50_checkpoint),
        "sidecar_checkpoint_sha256": sha(sidecar_checkpoint),
        "sidecar_training_seal_sha256": sha(sidecar_seal),
        "sidecar_training_freeze_sha256": sha(sidecar_training_freeze),
        "contract_sha256": sha(CONTRACT),
    }
    for key, digest in expected.items():
        if value.get(key) != digest:
            raise ValueError(f"FullHistory evaluation input changed: {key}")
    for name, digest in value["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"FullHistory evaluation source changed: {name}")
    _verify_sidecar(sidecar_checkpoint, sidecar_seal, sidecar_training_freeze,
                    main_checkpoint, dataset_path)
    return value


def _transition(episode: dict, step: int, device) -> PublicTransition:
    phase = torch.zeros((), dtype=torch.long, device=device)
    return PublicTransition(
        _to(episode["u"][step], device), _to(episode["q"][step], device),
        _to(episode["executed_world_delta6"][step], device),
        _to(episode["u"][step + 1], device), _to(episode["q"][step + 1], device),
        phase, phase.clone(), step, step + 1, episode["episode_id"])


def _query(core, reference, episode, frame_row, step, scope, matched_hidden,
           wrong_hidden, initial_z, fixed_geometry, fixed_actions,
           compute_sensitivity):
    device = initial_z.device
    u = _to(episode["readout_u"][frame_row], device)
    with torch.no_grad():
        matched_z = reference.correction(matched_hidden)
        wrong_z = reference.correction(wrong_hidden)
        matched_geometry = core.geometry(u, matched_z)
        wrong_geometry = core.geometry(u, wrong_z)
    rows = []
    for method, geometry, latent in (
        ("full_history", matched_geometry, matched_z),
        ("wrong_full_history", wrong_geometry, wrong_z),
    ):
        rows.append(_query_record(
            core, episode, frame_row, step, scope, method, geometry, latent,
            initial_z, fixed_geometry, fixed_actions, inner_loss=None,
            compute_sensitivity=compute_sensitivity))
    return rows


def evaluate_full_history(core, reference, dataset: dict, *, compute_sensitivity=True) -> dict:
    """Replay matched and fixed cyclic-donor full histories without physics."""
    _validate_evaluation_fields(dataset)
    device = next(core.parameters()).device
    records = []
    for split, episodes in dataset["splits"].items():
        identities = [episode["episode_id"] for episode in episodes]
        strata = [episode["distribution"] for episode in episodes]
        wrong_map = cyclic_wrong_history_map(identities, strata)
        by_id = {episode["episode_id"]: episode for episode in episodes}
        for episode in episodes:
            donor = by_id[wrong_map[episode["episode_id"]]]
            first_u = _to(episode["u"][0], device)
            with torch.no_grad():
                matched_hidden = reference.initial_hidden(core, first_u)
                # Deliberately keep recipient U0; only transition identity is wrong.
                wrong_hidden = reference.initial_hidden(core, first_u)
                initial_z = reference.correction(matched_hidden)
            for step in range(25):
                if step:
                    with torch.no_grad():
                        matched_hidden = reference.advance(
                            core, matched_hidden, _transition(episode, step - 1, device))
                        wrong_hidden = reference.advance(
                            core, wrong_hidden, _transition(donor, step - 1, device))
                u = _to(episode["readout_u"][step], device)
                q = _to(episode["readout_q"][step], device)
                with torch.no_grad():
                    fixed_geometry = core.geometry(u, initial_z)
                    fixed_actions = core.p50(
                        fixed_geometry.expand(4, 12), q.expand(4, 9),
                        torch.arange(4, device=device))
                rows = _query(core, reference, episode, step, step, "continuous",
                              matched_hidden, wrong_hidden, initial_z, fixed_geometry,
                              fixed_actions, compute_sensitivity)
                for row in rows:
                    row["split"] = split
                records.extend(rows)
            steps = episode["readout_step_index"].tolist()
            for future_step in FUTURE_STEPS:
                frame_row = steps.index(future_step)
                u = _to(episode["readout_u"][frame_row], device)
                q = _to(episode["readout_q"][frame_row], device)
                with torch.no_grad():
                    fixed_geometry = core.geometry(u, initial_z)
                    fixed_actions = core.p50(
                        fixed_geometry.expand(4, 12), q.expand(4, 9),
                        torch.arange(4, device=device))
                rows = _query(core, reference, episode, frame_row, future_step,
                              "future_no_adaptation", matched_hidden, wrong_hidden,
                              initial_z, fixed_geometry, fixed_actions, compute_sensitivity)
                for row in rows:
                    row["split"] = split
                records.extend(rows)
    return {
        "schema": "prove.p59-full-history-offline-evaluation.v1",
        "methods": ["full_history", "wrong_full_history"],
        "wrong_context_contract": "same recipient U0 initialization; fixed cyclic donor actual transitions from t0 onward",
        "future_query_contract": "t27/t29 retain the hidden state after t24; sparse gaps never update it",
        "physics_rollouts": 0, "records": records, "aggregate": _aggregate(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--main-checkpoint", type=Path, required=True)
    parser.add_argument("--main-seal", type=Path, required=True)
    parser.add_argument("--main-training-freeze", type=Path, required=True)
    parser.add_argument("--p50-checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar-checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar-seal", type=Path, required=True)
    parser.add_argument("--sidecar-training-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    common = (args.dataset, args.main_checkpoint, args.main_seal,
              args.main_training_freeze, args.p50_checkpoint,
              args.sidecar_checkpoint, args.sidecar_seal,
              args.sidecar_training_freeze, args.output)
    if args.mode == "prepare":
        prepare_evaluation(*common)
        return
    freeze = verify_evaluation_freeze(*common)
    core, _, _ = load_sealed_candidate(
        args.main_checkpoint, args.main_seal, args.main_training_freeze,
        args.p50_checkpoint, device=args.device)
    reference = load_full_history_checkpoint(
        args.sidecar_checkpoint, args.main_checkpoint, device=args.device)
    dataset = load_dataset(args.dataset, core.contract.u_dim)
    result = evaluate_full_history(
        core, reference, dataset, compute_sensitivity=not args.skip_sensitivity)
    result["provenance"] = {"evaluation_freeze_sha256": sha(
        args.output / "FULL_HISTORY_EVAL_FREEZE.json"), **freeze}
    with (args.output / "FULL_HISTORY_EVALUATION.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
