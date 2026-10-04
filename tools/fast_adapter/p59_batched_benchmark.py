#!/usr/bin/env python3
"""Train-only equivalence and throughput audit for P59 batched history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from prove.task_graph.p59 import P59Contract, PublicTransition
from prove.task_graph.p59.core import P59AdaptationCore
from prove.task_graph.p59.core_batched import BatchedP59AdaptationCore
from prove.task_graph.p59.train import _batched_support, _stack_frames


ROOT = Path(__file__).resolve().parents[2]
P50 = ROOT / "evaluations/fast_adapter_p50_precision_policy_v01/offline_v01/precision_action.pt"
READOUT = ROOT / "evaluations/fast_adapter_p59_control_latent_v01/speed_check_v01/server_public_readout.pt"


def build_pair(dataset, dtype, device):
    torch.manual_seed(11)
    original = P59AdaptationCore(P59Contract(u_dim=244), P50).to(device=device, dtype=dtype)
    readout = torch.load(READOUT, map_location="cpu", weights_only=True)
    if readout.get("schema") != "prove.p59-frozen-public-readout.v1" or readout.get("updates") != 1000:
        raise ValueError("benchmark requires the frozen train-only readout1000 state")
    original.public_readout.load_state_dict(readout["state_dict"], strict=True)
    original.set_public_normalization(readout["u_mean"].to(device=device, dtype=dtype),
                                      readout["u_scale"].to(device=device, dtype=dtype))
    fast = BatchedP59AdaptationCore(P59Contract(u_dim=244), P50).to(device=device, dtype=dtype)
    fast.load_state_dict(original.state_dict(), strict=True)
    original.freeze_public_readout()
    fast.freeze_public_readout()
    return original, fast


def real_case(dataset, steps, dtype, device):
    episodes = dataset["splits"]["train"]
    indices = torch.arange(16)
    support = _batched_support(episodes, indices, steps, device)
    if dtype != torch.float32:
        support = [PublicTransition(item.u.to(dtype), item.q.to(dtype), item.executed_action.to(dtype),
                    item.next_u.to(dtype), item.next_q.to(dtype), item.phase, item.next_phase,
                    item.step_index, item.next_step_index, item.episode_id) for item in support]
    query_indices = torch.full((16,), steps)
    u = _stack_frames(episodes, indices, query_indices, "u", device).to(dtype)
    q = _stack_frames(episodes, indices, query_indices, "q", device).to(dtype)
    geometry = _stack_frames(episodes, indices, query_indices, "oracle_geometry", device).to(dtype)
    all_actions = _stack_frames(episodes, indices, query_indices, "oracle_p50_action6", device).to(dtype)
    phase = torch.arange(16, device=device) % 4
    action = all_actions[torch.arange(16, device=device), phase]
    return support, u, q, phase, action, geometry


def synthetic_case(steps, dtype, device):
    generator = torch.Generator().manual_seed(5902)
    readout = torch.load(READOUT, map_location="cpu", weights_only=True)
    u_mean = readout["u_mean"].to(dtype)
    u_scale = readout["u_scale"].to(dtype)
    q_base = torch.tensor([.1, .2, .3, 1., 0., 0., 0., 1., 0.], dtype=dtype)
    u = [(u_mean + u_scale * torch.randn(16, 244, generator=generator, dtype=dtype)).to(device)
         for _ in range(steps + 1)]
    q = [(q_base.expand(16, -1) + .001 * torch.randn(16, 9, generator=generator, dtype=dtype)).to(device)
         for _ in range(steps + 1)]
    phase0 = torch.zeros(16, dtype=torch.long, device=device)
    rows = [PublicTransition(u[t], q[t], (.01 * torch.randn(16, 6, generator=generator, dtype=dtype)).to(device),
                             u[t + 1], q[t + 1], phase0, phase0, t, t + 1, "synthetic")
            for t in range(steps)]
    phase = torch.arange(16, device=device) % 4
    # Oracle values are only stable finite timing targets here.
    geometry = torch.cat((torch.zeros(16, 3, dtype=dtype, device=device),
                          torch.tensor([1., 0., 0., 0., 1., 0.], dtype=dtype, device=device).expand(16, -1),
                          torch.full((16, 2), .015, dtype=dtype, device=device),
                          torch.full((16, 1), .05, dtype=dtype, device=device)), -1)
    action = torch.zeros(16, 6, dtype=dtype, device=device)
    return rows, u[-1], q[-1], phase, action, geometry


def trainable(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if name.startswith("initializer.") or name.startswith("feedback_head.")]


def outer(model, case):
    support, u, q, phase, action, geometry = case
    return model.outer_loss(support, u, q, phase, action, geometry,
                            query_step_index=len(support), query_episode_id=support[0].episode_id)


def equivalence(dataset, device):
    results = {}
    for label, dtype, tolerance in (("float64", torch.float64, 2e-10),
                                    ("float32", torch.float32, 5e-5)):
        original, fast = build_pair(dataset, dtype, device)
        case = real_case(dataset, 12, dtype, device)
        loss_a, parts_a = outer(original, case)
        loss_b, parts_b = outer(fast, case)
        params_a, params_b = trainable(original), trainable(fast)
        grads_a = torch.autograd.grad(loss_a, [value for _, value in params_a], allow_unused=True)
        grads_b = torch.autograd.grad(loss_b, [value for _, value in params_b], allow_unused=True)
        grad_max = max(float((a - b).abs().max()) for a, b in zip(grads_a, grads_b)
                       if a is not None and b is not None)
        results[label] = {
            "loss_abs_difference": float((loss_a - loss_b).abs()),
            "z_max_abs_difference": float((parts_a["adapted_z"] - parts_b["adapted_z"]).abs().max()),
            "action_max_abs_difference": float((parts_a["online_action"] - parts_b["online_action"]).abs().max()),
            "all_trainable_gradient_max_abs_difference": grad_max,
            "passed": all(value <= tolerance for value in (
                float((loss_a - loss_b).abs()),
                float((parts_a["adapted_z"] - parts_b["adapted_z"]).abs().max()), grad_max)),
        }
    return results


def timed(model, case, repeats=3):
    parameters = [value for _, value in trainable(model)]
    values = []
    for _ in range(repeats + 1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss, _ = outer(model, case)
        torch.autograd.grad(loss, parameters, allow_unused=True)
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
    return values[1:]


def throughput(dataset, device):
    original, fast = build_pair(dataset, torch.float32, device)
    output = {}
    for source in ("real_train", "synthetic"):
        output[source] = {}
        for steps in (12, 24):
            case = (real_case(dataset, steps, torch.float32, device) if source == "real_train"
                    else synthetic_case(steps, torch.float32, device))
            slow, quick = timed(original, case), timed(fast, case)
            output[source][f"q{steps}"] = {
                "original_seconds": slow,
                "batched_seconds": quick,
                "speedup_mean": (sum(slow) / len(slow)) / (sum(quick) / len(quick)),
            }
    return output


def short_training(dataset, device):
    original, fast = build_pair(dataset, torch.float32, device)
    case = real_case(dataset, 12, torch.float32, device)
    pa, pb = trainable(original), trainable(fast)
    oa = torch.optim.Adam([value for _, value in pa], lr=3e-4)
    ob = torch.optim.Adam([value for _, value in pb], lr=3e-4)
    losses = []
    for _ in range(10):
        oa.zero_grad(set_to_none=True)
        ob.zero_grad(set_to_none=True)
        la, _ = outer(original, case)
        lb, _ = outer(fast, case)
        la.backward()
        lb.backward()
        oa.step()
        ob.step()
        losses.append([float(la.detach()), float(lb.detach())])
    return {
        "steps": 10,
        "final_loss_abs_difference": abs(losses[-1][0] - losses[-1][1]),
        "max_parameter_abs_difference": max(float((a - b).abs().max())
                                             for (_, a), (_, b) in zip(pa, pb)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("benchmark output exists")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    if dataset.get("schema") != "prove.p59-training-dataset.v1":
        raise ValueError("wrong dataset")
    result = {
        "schema": "prove.p59-batched-backend-benchmark.v1",
        "scope": "train split and synthetic tensors only; no developer outcomes or checkpoint selection",
        "dataset_sha256_expected": "a101b767aa4f99214e2fae6ffe6dc0d272a0fa15eaac284c9b658bbb8e7b01c1",
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "equivalence": equivalence(dataset, args.device),
        "throughput": throughput(dataset, args.device),
        "short_training": short_training(dataset, args.device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
