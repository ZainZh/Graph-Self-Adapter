from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from prove.fast_adapter import (
    AdaptationLossConfig,
    FastAdapter,
    FastAdapterSystem,
    HistoryConditionedAdapter,
    InteractionEncoder,
    ScriptedNominalPolicy2D,
    DynamicsPredictor,
    Toy2DConfig,
    Toy2DVectorEnv,
    expand_latent,
    inner_update_z,
    sample_toy2d_domains,
)


DEFAULT_K_VALUES = [0, 1, 2, 3, 5]


def build_system(config: Toy2DConfig, *, device: torch.device, inner_lr: float = 0.05) -> FastAdapterSystem:
    system = FastAdapterSystem(
        encoder=InteractionEncoder(obj_dim=2, tar_dim=2, proprio_dim=2, action_dim=2),
        base_policy=ScriptedNominalPolicy2D(action_limit=config.action_limit),
        adapter=FastAdapter(max_residual=torch.tensor([0.15, 0.15])),
        dynamics=DynamicsPredictor(),
        inner_lr=inner_lr,
    )
    return system.to(device)


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    sq = 0.0
    for param in parameters:
        if param.grad is not None:
            sq += float(param.grad.detach().square().sum().cpu())
    return math.sqrt(sq)


def support_adapt(
    system: FastAdapterSystem,
    env: Toy2DVectorEnv,
    domain,
    obs,
    *,
    k: int,
    loss_config: AdaptationLossConfig,
    create_graph: bool = True,
) -> tuple[torch.Tensor, object, Dict[str, float]]:
    z = expand_latent(system.z0, domain.gain.shape[0])
    losses = []
    for _ in range(k):
        action, _delta, _base, u = system.act(obs.adapter_observation(), z)
        next_obs = env.step(obs, domain, action)
        u_next = system.encode(next_obs.adapter_observation())
        update = inner_update_z(
            system=system,
            u=u,
            action_observed=next_obs.prev_action,
            z=z,
            u_next=u_next,
            loss_config=loss_config,
            create_graph=create_graph,
            detach_executed_action=True,
        )
        losses.append(float(update.loss.detach().cpu()) / domain.gain.shape[0])
        z = update.z_next
        obs = next_obs
    latent_delta = torch.linalg.norm(z.detach() - system.z0.detach().unsqueeze(0), dim=-1).mean()
    return z, obs, {
        "support_loss": sum(losses) / max(1, len(losses)),
        "latent_delta_norm": float(latent_delta.cpu()),
    }


def query_loss(
    system: FastAdapterSystem,
    env: Toy2DVectorEnv,
    domain,
    obs,
    z: torch.Tensor,
    *,
    horizon: int,
    residual_reg: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    action_losses = []
    residual_losses = []
    goal_errors = []
    success = None
    first_action_mse = None
    for step in range(horizon):
        action, delta, _base, _u = system.act(obs.adapter_observation(), z)
        expert = env.expert_action(obs, domain)
        executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
        mse = (executed - expert).square().mean(dim=-1)
        next_obs = env.step(obs, domain, executed.detach())
        if step == 0:
            first_action_mse = float(mse.mean().detach().cpu())
        action_losses.append(mse.mean())
        residual_losses.append(delta.square().mean())
        goal_errors.append(env.goal_error(next_obs).detach())
        obs = next_obs
    final_error = goal_errors[-1]
    success = final_error <= env.config.success_threshold
    loss = torch.stack(action_losses).mean() + residual_reg * torch.stack(residual_losses).mean()
    metrics = {
        "query_action_mse": float(torch.stack([x.detach() for x in action_losses]).mean().cpu()),
        "first_query_action_mse": float(first_action_mse),
        "final_goal_error": float(final_error.mean().cpu()),
        "success_rate": float(success.float().mean().cpu()),
    }
    return loss, metrics


def train_ours(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.train_seed)
    cfg = Toy2DConfig()
    env = Toy2DVectorEnv(cfg, device=device)
    system = build_system(cfg, device=device, inner_lr=args.inner_lr)
    opt = torch.optim.Adam(
        [p for p in system.parameters() if p.requires_grad],
        lr=args.lr,
    )
    loss_cfg = AdaptationLossConfig(lambda_cos=args.lambda_cos, lambda_z=args.lambda_z)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    config = {
        "mode": "ours_smoke",
        "train_seed": args.train_seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "support_k": args.support_k,
        "query_horizon": args.query_horizon,
        "lr": args.lr,
        "inner_lr": float(system.inner_lr.detach().cpu()),
        "lambda_cos": args.lambda_cos,
        "lambda_z": args.lambda_z,
        "residual_reg": args.residual_reg,
        "toy_config": cfg.to_dict(),
        "runtime_notes": {
            "outer_control_loss": "privileged expert action imitation on executed clipped action; goal_error is measurement only",
            "inner_action_observation": "executed clipped action next_obs.prev_action is detached before dynamics prediction",
            "query_adaptation": "z is frozen during query rollout",
            "hidden_domain_inputs": "gain/bias are used only by toy env and expert, never runtime adapter observation",
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    records = []
    started = time.perf_counter()
    with history_path.open("w") as handle:
        for update_idx in range(args.updates):
            step_start = time.perf_counter()
            domain = sample_toy2d_domains(
                args.batch_size,
                args.train_seed * 100000 + update_idx,
                cfg,
                device=device,
            )
            support_obs = env.reset(args.batch_size, args.train_seed * 200000 + update_idx, domain=domain)
            z, _support_tail, support_metrics = support_adapt(
                system,
                env,
                domain,
                support_obs,
                k=args.support_k,
                loss_config=loss_cfg,
            )
            query_obs = env.reset(args.batch_size, args.train_seed * 300000 + update_idx, domain=domain)
            loss, query_metrics = query_loss(
                system,
                env,
                domain,
                query_obs,
                z,
                horizon=args.query_horizon,
                residual_reg=args.residual_reg,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = {
                "encoder": grad_norm(system.encoder.parameters()),
                "adapter": grad_norm(system.adapter.parameters()),
                "dynamics": grad_norm(system.dynamics.parameters()),
                "z0": grad_norm([system.z0]),
            }
            opt.step()
            rec = {
                "update": update_idx + 1,
                "loss": float(loss.detach().cpu()),
                "sec_update": time.perf_counter() - step_start,
                **support_metrics,
                **query_metrics,
                "grad_norms": grads,
            }
            records.append(rec)
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            handle.flush()
            if args.log_every and ((update_idx + 1) % args.log_every == 0 or update_idx == 0):
                print(json.dumps(rec, sort_keys=True))
    elapsed = time.perf_counter() - started
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save({"system": system.state_dict(), "config": config, "history_tail": records[-10:]}, ckpt_path)
    summary = {
        "updates": args.updates,
        "elapsed_sec": elapsed,
        "mean_sec_update": elapsed / max(1, args.updates),
        "last": records[-1] if records else None,
        "checkpoint": str(ckpt_path),
        "history": str(history_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/smoke the PROVE Fast Adapter toy 2D prototype.")
    parser.add_argument("--output-dir", default="evaluations/fast_adapter_v01/smoke")
    parser.add_argument("--train-seed", type=int, default=11)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-k", type=int, default=3)
    parser.add_argument("--query-horizon", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--lambda-cos", type=float, default=0.1)
    parser.add_argument("--lambda-z", type=float, default=1.0e-4)
    parser.add_argument("--residual-reg", type=float, default=1.0e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    result = train_ours(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
