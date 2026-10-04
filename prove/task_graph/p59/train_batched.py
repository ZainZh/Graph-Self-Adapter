"""Thin P59 training wrapper selecting the equivalent batched backend."""
from __future__ import annotations

import json
from pathlib import Path

from .core_batched import BatchedP59AdaptationCore
from . import train as base


ROOT = Path(__file__).resolve().parents[3]
BASE_CONTRACT = ROOT / "evaluations/fast_adapter_p59_control_latent_v01/MODEL_CONTRACT.json"
EXECUTION_CONTRACT = ROOT / "evaluations/fast_adapter_p59_control_latent_v01/EXECUTION_CONTRACT_BATCHED.json"


def load_execution_contract() -> dict:
    model = json.loads(BASE_CONTRACT.read_text())
    execution = json.loads(EXECUTION_CONTRACT.read_text())
    if model.get("schema") != "prove.p59-model-contract.v1":
        raise ValueError("unknown base P59 model contract")
    if execution.get("schema") != "prove.p59-batched-execution-contract.v1":
        raise ValueError("unknown P59 execution contract")
    if base.sha(BASE_CONTRACT) != execution["base_model_contract_sha256"]:
        raise ValueError("base P59 model contract changed after backend review")
    model = json.loads(json.dumps(model))
    model["training"]["device"] = execution["device"]
    model["execution_backend"] = execution
    return model


def main() -> None:
    # train.py resolves these globals at call time. No model/loss/data logic is
    # copied, and the subclass has no parameters, so checkpoints stay compatible.
    base.P59AdaptationCore = BatchedP59AdaptationCore
    base.MODEL_CONTRACT = EXECUTION_CONTRACT
    base.load_contract = load_execution_contract
    base.SOURCES = list(base.SOURCES) + [
        "prove/task_graph/p59/core_batched.py",
        "prove/task_graph/p59/train_batched.py",
        "tests/test_p59_batched_core.py",
        "tools/fast_adapter/p59_batched_benchmark.py",
        "evaluations/fast_adapter_p59_control_latent_v01/EXECUTION_CONTRACT_BATCHED.json",
        "evaluations/fast_adapter_p59_control_latent_v01/EXECUTION_AMENDMENT_BATCHED.md",
        "evaluations/fast_adapter_p59_control_latent_v01/speed_check_v01/BATCHED_BACKEND_BENCHMARK.json",
    ]
    base.main()


if __name__ == "__main__":
    main()
