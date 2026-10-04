from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from prove.fast_adapter import (
    AdaptationLossConfig,
    FastAdapter,
    FastAdapterSystem,
    InteractionEncoder,
    ScriptedNominalPolicy2D,
    DynamicsPredictor,
    Toy2DConfig,
    Toy2DDomain,
    Toy2DObservation,
    Toy2DVectorEnv,
    expand_latent,
    sample_toy2d_domains,
)
from tools.fast_adapter.p15_experiment import (
    P15HistorySystem,
    HistoryConfig,
    TransitionBatch,
    count_parameters,
    grad_norm,
    repeat_domain,
    roll1_derangement,
    sha256_file,
    tensor_state_hash,
)

P16_ROOT = Path("evaluations/fast_adapter_p16_v01")
TRAIN_SEEDS_DEV = [11, 22, 33]
TRAIN_SEEDS_FINAL_EXTRA = [44, 55]
FAST_CELLS = [(3, 0), (3, 1), (3, 3), (3, 5), (1, 3), (5, 3)]
HISTORY_S_VALUES = [0, 1, 3, 5]
SUPPORT_MAX_S = 5
QUERY_HORIZON = 10
QUERY_REPEATS = 8


@dataclass(frozen=True)
class P16FastConfig:
    config_id: str
    method: Literal["feature_fast", "displacement_fast"]
    z_dim: int = 32
    inner_lr_init: float = 0.5
    updates: int = 1500
    batch_size: int = 64
    support_s_train: int = 3
    gradient_u_train: int = 3
    query_horizon: int = 10
    lr: float = 1.0e-3
    lambda_cos: float = 0.1
    lambda_z: float = 1.0e-4
    residual_reg: float = 1.0e-3
    physical_outer_weight: float = 1.0
    train_stream_offset: int = 25_000_000


@dataclass(frozen=True)
class P16HistoryConfig:
    config_id: str = "history"
    width: int = 256
    updates: int = 1500
    batch_size: int = 64
    support_s_train: int = 3
    query_horizon: int = 10
    lr: float = 1.0e-3
    residual_reg: float = 1.0e-3
    train_stream_offset: int = 25_000_000


@dataclass(frozen=True)
class SupportRecord:
    u: Tensor
    action: Tensor
    u_next: Tensor
    state: Tensor
    next_state: Tensor

    @property
    def delta_x(self) -> Tensor:
        return (self.next_state - self.state).detach()

    def as_transition(self) -> TransitionBatch:
        return TransitionBatch(u=self.u, action=self.action, u_next=self.u_next)


class PhysicalDisplacementPredictor(nn.Module):
    def __init__(self, u_dim: int = 256, z_dim: int = 32, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_dim + action_dim + z_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 2),
        )

    def forward(self, u: Tensor, action: Tensor, z: Tensor) -> Tensor:
        return self.net(torch.cat([u, action, z], dim=-1))


def fast_configs() -> List[P16FastConfig]:
    return [
        P16FastConfig(config_id="feature_fast", method="feature_fast"),
        P16FastConfig(config_id="displacement_fast", method="displacement_fast"),
    ]


def history_config() -> P16HistoryConfig:
    return P16HistoryConfig()


def build_feature_fast_system(config: P16FastConfig, *, device: torch.device) -> FastAdapterSystem:
    toy = Toy2DConfig()
    system = FastAdapterSystem(
        encoder=InteractionEncoder(obj_dim=2, tar_dim=2, proprio_dim=2, action_dim=2),
        base_policy=ScriptedNominalPolicy2D(action_limit=toy.action_limit),
        adapter=FastAdapter(z_dim=config.z_dim, max_residual=torch.tensor([0.15, 0.15])),
        dynamics=DynamicsPredictor(z_dim=config.z_dim),
        z_dim=config.z_dim,
        learn_inner_lr=True,
        inner_lr=config.inner_lr_init,
    )
    return system.to(device)


def build_displacement_fast_system(config: P16FastConfig, *, device: torch.device) -> FastAdapterSystem:
    system = build_feature_fast_system(config, device=device)
    system.dynamics = PhysicalDisplacementPredictor(z_dim=config.z_dim).to(device)  # type: ignore[assignment]
    return system


def build_fast_system(config: P16FastConfig, *, device: torch.device) -> FastAdapterSystem:
    if config.method == "feature_fast":
        return build_feature_fast_system(config, device=device)
    if config.method == "displacement_fast":
        return build_displacement_fast_system(config, device=device)
    raise ValueError(config.method)


def build_history_system(config: P16HistoryConfig, *, device: torch.device) -> P15HistorySystem:
    return P15HistorySystem(HistoryConfig(config_id=config.config_id, width=config.width)).to(device)


def collect_common_base_records(system: nn.Module, env: Toy2DVectorEnv, domain: Toy2DDomain, *, support_seed: int, max_s: int) -> List[SupportRecord]:
    obs = env.reset(domain.gain.shape[0], support_seed, domain=domain)
    records: List[SupportRecord] = []
    for _ in range(max_s):
        if isinstance(system, P15HistorySystem):
            u = system.encode(obs)
            base = system.base_action(obs, u)
        else:
            u = system.encode(obs.adapter_observation())
            base = system.base_action(obs.adapter_observation(), u)
        executed = torch.clamp(base, -env.config.action_limit, env.config.action_limit)
        next_obs = env.step(obs, domain, executed.detach())
        if isinstance(system, P15HistorySystem):
            u_next = system.encode(next_obs)
        else:
            u_next = system.encode(next_obs.adapter_observation())
        records.append(SupportRecord(u=u, action=next_obs.prev_action.detach(), u_next=u_next, state=obs.state.detach(), next_state=next_obs.state.detach()))
        obs = next_obs
    return records


def feature_terms(system: FastAdapterSystem, record: SupportRecord, z: Tensor, config: P16FastConfig) -> Dict[str, Tensor]:
    pred = system.dynamics(record.u, record.action.detach(), z)
    target = record.u_next.detach()
    mse = (pred - target).square().mean(dim=-1)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1)
    z_ref = system.z0.unsqueeze(0).expand_as(z)
    z_reg = (z - z_ref).square().mean(dim=-1)
    total = mse + config.lambda_cos * cos + config.lambda_z * z_reg
    return {"mse": mse, "cos": cos, "z_reg": z_reg, "total": total}


def displacement_terms(system: FastAdapterSystem, record: SupportRecord, z: Tensor, config: P16FastConfig) -> Dict[str, Tensor]:
    pred = system.dynamics(record.u, record.action.detach(), z)
    target = record.delta_x
    mse = (pred - target).square().mean(dim=-1)
    z_ref = system.z0.unsqueeze(0).expand_as(z)
    z_reg = (z - z_ref).square().mean(dim=-1)
    total = mse + config.lambda_z * z_reg
    return {"mse": mse, "z_reg": z_reg, "total": total}


def support_loss_per_domain(system: FastAdapterSystem, records: Sequence[SupportRecord], z: Tensor, config: P16FastConfig) -> Dict[str, Tensor]:
    if not records:
        zeros = torch.zeros(z.shape[0], device=z.device)
        out = {"mse": zeros, "z_reg": zeros, "total": zeros}
        if config.method == "feature_fast":
            out["cos"] = zeros
        return out
    sums: Dict[str, Tensor] = {}
    for record in records:
        terms = feature_terms(system, record, z, config) if config.method == "feature_fast" else displacement_terms(system, record, z, config)
        for key, value in terms.items():
            sums[key] = sums.get(key, torch.zeros_like(value)) + value
    return {key: value / float(len(records)) for key, value in sums.items()}


def terms_to_float(prefix: str, terms: Dict[str, Tensor]) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value.detach().mean().cpu()) for key, value in terms.items()}


def inner_update_full_batch(system: FastAdapterSystem, records: Sequence[SupportRecord], z: Tensor, config: P16FastConfig, *, create_graph: bool) -> tuple[Tensor, Dict[str, float]]:
    pre = support_loss_per_domain(system, records, z, config)
    loss = pre["total"].sum()
    grad_z = torch.autograd.grad(loss, z, create_graph=create_graph, retain_graph=create_graph, only_inputs=True)[0]
    z_next = z - system.inner_lr * grad_z
    z_next.retain_grad()
    post = support_loss_per_domain(system, records, z_next, config)
    metrics = terms_to_float("pre", pre)
    metrics.update(terms_to_float("post", post))
    metrics["grad_z_norm"] = float(torch.linalg.norm(grad_z.detach(), dim=-1).mean().cpu())
    return z_next, metrics


def adapt_fast(system: FastAdapterSystem, records: Sequence[SupportRecord], *, s: int, u_passes: int, config: P16FastConfig, create_graph: bool) -> tuple[Tensor, List[Dict[str, float]]]:
    if s < 0 or u_passes < 0:
        raise ValueError("S and U must be non-negative")
    if s > len(records):
        raise ValueError(f"requested S={s} but only {len(records)} records are available")
    z = expand_latent(system.z0, records[0].u.shape[0] if records else 1)
    if not records:
        raise ValueError("records must include at least one batch record for batch size")
    support = list(records[:s])
    metrics: List[Dict[str, float]] = []
    for pass_index in range(u_passes):
        if not support:
            metrics.append({"pass_index": pass_index, "support_records": 0, "grad_z_norm": 0.0})
            continue
        z, step = inner_update_full_batch(system, support, z, config, create_graph=create_graph)
        step.update({"pass_index": pass_index, "support_records": s})
        metrics.append(step)
    return z, metrics


def support_displacement_outer_mse(system: FastAdapterSystem, records: Sequence[SupportRecord], z: Tensor) -> Tensor:
    losses = []
    for record in records:
        pred = system.dynamics(record.u, record.action.detach(), z)
        losses.append((pred - record.delta_x).square().mean(dim=-1).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=z.device)


def query_loss_fast(system: FastAdapterSystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, z: Tensor, *, horizon: int, residual_reg: float) -> tuple[Tensor, Dict[str, float]]:
    action_losses = []
    residual_losses = []
    first_mse = None
    for _ in range(horizon):
        action, delta, _base, _u = system.act(obs.adapter_observation(), z)
        executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
        expert = env.expert_action(obs, domain)
        mse = (executed - expert).square().mean(dim=-1)
        if first_mse is None:
            first_mse = float(mse.mean().detach().cpu())
        action_losses.append(mse.mean())
        residual_losses.append(delta.square().mean())
        obs = env.step(obs, domain, executed.detach())
    final_error = env.goal_error(obs).detach()
    success = env.success(obs).detach()
    loss = torch.stack(action_losses).mean() + residual_reg * torch.stack(residual_losses).mean()
    return loss, {"first_query_action_mse": float(first_mse), "query_action_mse": float(torch.stack([x.detach() for x in action_losses]).mean().cpu()), "final_goal_error": float(final_error.mean().cpu()), "success_rate": float(success.float().mean().cpu())}


def history_update_context(system: P15HistorySystem, records: Sequence[SupportRecord], *, s: int, batch_size: int, device: torch.device) -> Tensor:
    context = system.adapter.initial_context(batch_size, device=device)
    for record in records[:s]:
        context = system.adapter.update_context(context, record.u, record.action, record.u_next)
    return context


def query_loss_history(system: P15HistorySystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, context: Tensor, *, horizon: int, residual_reg: float) -> tuple[Tensor, Dict[str, float]]:
    action_losses = []
    residual_losses = []
    first_mse = None
    for _ in range(horizon):
        u = system.encode(obs)
        base = system.base_action(obs, u)
        delta = system.adapter(u, context, base)
        executed = torch.clamp(base + delta, -env.config.action_limit, env.config.action_limit)
        expert = env.expert_action(obs, domain)
        mse = (executed - expert).square().mean(dim=-1)
        if first_mse is None:
            first_mse = float(mse.mean().detach().cpu())
        action_losses.append(mse.mean())
        residual_losses.append(delta.square().mean())
        obs = env.step(obs, domain, executed.detach())
    final_error = env.goal_error(obs).detach()
    success = env.success(obs).detach()
    loss = torch.stack(action_losses).mean() + residual_reg * torch.stack(residual_losses).mean()
    return loss, {"first_query_action_mse": float(first_mse), "query_action_mse": float(torch.stack([x.detach() for x in action_losses]).mean().cpu()), "final_goal_error": float(final_error.mean().cpu()), "success_rate": float(success.float().mean().cpu())}


def train_fast(args: argparse.Namespace) -> Dict[str, object]:
    cfg = P16FastConfig(config_id=args.config_id, method=args.method, updates=args.updates, batch_size=args.batch_size, support_s_train=args.support_s_train, gradient_u_train=args.gradient_u_train, query_horizon=args.query_horizon, train_stream_offset=args.train_stream_offset)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.train_seed)
    toy = Toy2DConfig()
    env = Toy2DVectorEnv(toy, device=device)
    system = build_fast_system(cfg, device=device)
    opt = torch.optim.Adam([p for p in system.parameters() if p.requires_grad], lr=cfg.lr)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    config = {"mode": "p16_fast", "train_seed": args.train_seed, "fast_config": asdict(cfg), "toy_config": toy.to_dict(), "param_count": count_parameters(system)}
    (outdir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    records = []
    started = time.perf_counter()
    with (outdir / "history.jsonl").open("w") as handle:
        for update_idx in range(cfg.updates):
            step_start = time.perf_counter()
            stream = cfg.train_stream_offset + args.train_seed * 100000 + update_idx
            domain = sample_toy2d_domains(cfg.batch_size, stream, toy, device=device)
            support_records = collect_common_base_records(system, env, domain, support_seed=stream + 10000000, max_s=cfg.support_s_train)
            z, support_metrics = adapt_fast(system, support_records, s=cfg.support_s_train, u_passes=cfg.gradient_u_train, config=cfg, create_graph=True)
            query_obs = env.reset(cfg.batch_size, stream + 20000000, domain=domain)
            loss, query_metrics = query_loss_fast(system, env, domain, query_obs, z, horizon=cfg.query_horizon, residual_reg=cfg.residual_reg)
            physical_outer = torch.tensor(0.0, device=device)
            if cfg.method == "displacement_fast":
                physical_outer = support_displacement_outer_mse(system, support_records, z)
                loss = loss + cfg.physical_outer_weight * physical_outer
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = {"encoder": grad_norm(system.encoder.parameters()), "adapter": grad_norm(system.adapter.parameters()), "dynamics": grad_norm(system.dynamics.parameters()), "z0": grad_norm([system.z0]), "log_inner_lr": grad_norm([system.log_inner_lr]) if system.log_inner_lr is not None else 0.0}
            opt.step()
            finite = bool(torch.isfinite(loss.detach()).item() and torch.isfinite(system.inner_lr.detach()).item())
            rec = {"update": update_idx + 1, "loss": float(loss.detach().cpu()), "finite": finite, "inner_lr": float(system.inner_lr.detach().cpu()), "latent_delta_norm": float(torch.linalg.norm(z.detach() - system.z0.detach().unsqueeze(0), dim=-1).mean().cpu()), "physical_outer_mse": float(physical_outer.detach().cpu()), "sec_update": time.perf_counter() - step_start, "support_metrics": support_metrics, **query_metrics, "grad_norms": grads}
            records.append(rec); handle.write(json.dumps(rec, sort_keys=True) + "\n")
            if not finite:
                break
            if args.log_every and (update_idx == 0 or (update_idx + 1) % args.log_every == 0):
                print(json.dumps(rec, sort_keys=True))
    elapsed = time.perf_counter() - started
    ckpt_path = outdir / "checkpoint.pt"
    torch.save({"system": system.state_dict(), "config": config, "history_tail": records[-10:]}, ckpt_path)
    summary = {"checkpoint": str(ckpt_path), "elapsed_sec": elapsed, "mean_sec_update": elapsed / max(1, len(records)), "completed_updates": len(records), "final_parameter_allfinite": all(torch.isfinite(p.detach()).all().item() for p in system.parameters()), "last": records[-1] if records else None}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def train_history(args: argparse.Namespace) -> Dict[str, object]:
    cfg = P16HistoryConfig(updates=args.updates, batch_size=args.batch_size, support_s_train=args.support_s_train, query_horizon=args.query_horizon, train_stream_offset=args.train_stream_offset)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.train_seed)
    toy = Toy2DConfig(); env = Toy2DVectorEnv(toy, device=device)
    system = build_history_system(cfg, device=device)
    opt = torch.optim.Adam([p for p in system.parameters() if p.requires_grad], lr=cfg.lr)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    config = {"mode": "p16_history", "train_seed": args.train_seed, "history_config": asdict(cfg), "toy_config": toy.to_dict(), "param_count": count_parameters(system)}
    (outdir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    records=[]; started=time.perf_counter()
    with (outdir / "history.jsonl").open("w") as handle:
        for update_idx in range(cfg.updates):
            step_start=time.perf_counter()
            stream=cfg.train_stream_offset + args.train_seed * 100000 + update_idx
            domain=sample_toy2d_domains(cfg.batch_size, stream, toy, device=device)
            support_records=collect_common_base_records(system, env, domain, support_seed=stream + 10000000, max_s=cfg.support_s_train)
            context=history_update_context(system, support_records, s=cfg.support_s_train, batch_size=cfg.batch_size, device=device)
            query_obs=env.reset(cfg.batch_size, stream + 20000000, domain=domain)
            loss, query_metrics=query_loss_history(system, env, domain, query_obs, context, horizon=cfg.query_horizon, residual_reg=cfg.residual_reg)
            opt.zero_grad(set_to_none=True); loss.backward()
            grads={"encoder": grad_norm(system.encoder.parameters()), "adapter": grad_norm(system.adapter.parameters())}
            opt.step()
            finite=bool(torch.isfinite(loss.detach()).item())
            rec={"update":update_idx+1,"loss":float(loss.detach().cpu()),"finite":finite,"sec_update":time.perf_counter()-step_start,**query_metrics,"grad_norms":grads}
            records.append(rec); handle.write(json.dumps(rec, sort_keys=True)+"\n")
            if not finite: break
            if args.log_every and (update_idx == 0 or (update_idx + 1) % args.log_every == 0): print(json.dumps(rec, sort_keys=True))
    elapsed=time.perf_counter()-started
    ckpt_path=outdir/"checkpoint.pt"
    torch.save({"system": system.state_dict(), "config": config, "history_tail": records[-10:]}, ckpt_path)
    summary={"checkpoint":str(ckpt_path),"elapsed_sec":elapsed,"mean_sec_update":elapsed/max(1,len(records)),"completed_updates":len(records),"final_parameter_allfinite":all(torch.isfinite(p.detach()).all().item() for p in system.parameters()),"last":records[-1] if records else None}
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_fast(checkpoint: Path, *, device: torch.device) -> tuple[FastAdapterSystem, Dict[str, object], P16FastConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = P16FastConfig(**ckpt["config"]["fast_config"])
    system = build_fast_system(cfg, device=device)
    system.load_state_dict(ckpt["system"])
    system.eval()
    for p in system.parameters(): p.requires_grad_(False)
    return system, ckpt["config"], cfg


def load_history(checkpoint: Path, *, device: torch.device) -> tuple[P15HistorySystem, Dict[str, object], P16HistoryConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = P16HistoryConfig(**ckpt["config"]["history_config"])
    system = build_history_system(cfg, device=device)
    system.load_state_dict(ckpt["system"])
    system.eval()
    for p in system.parameters(): p.requires_grad_(False)
    return system, ckpt["config"], cfg


def rollout_fast_arrays(system: FastAdapterSystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, z: Tensor, z_frozen: Tensor, *, arm: str, horizon: int) -> Dict[str, Tensor]:
    first_mse=None; first_change=None; mse_steps=[]; delta_norm=torch.zeros(obs.state.shape[0], device=obs.state.device)
    with torch.no_grad():
        frozen_action, _fd, _fb, _fu = system.act(obs.adapter_observation(), z_frozen)
        frozen_executed = torch.clamp(frozen_action, -env.config.action_limit, env.config.action_limit)
        for step in range(horizon):
            if arm == "base":
                action = env.nominal_action(obs); delta_norm=torch.zeros(action.shape[0], device=action.device)
            else:
                action, delta, _base, _u = system.act(obs.adapter_observation(), z); delta_norm=torch.linalg.norm(delta, dim=-1)
            executed=torch.clamp(action, -env.config.action_limit, env.config.action_limit)
            expert=env.expert_action(obs, domain)
            mse=(executed-expert).square().mean(dim=-1)
            if first_mse is None:
                first_mse=mse; first_change=torch.linalg.norm(executed - frozen_executed, dim=-1)
            mse_steps.append(mse); obs=env.step(obs, domain, executed)
        return {"first_mse":first_mse,"first_action_change_vs_frozen":first_change,"query_mse":torch.stack(mse_steps).mean(dim=0),"final_error":env.goal_error(obs),"success":env.success(obs),"delta_norm":delta_norm}


def rollout_history_arrays(system: P15HistorySystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, context: Tensor, frozen_context: Tensor, *, horizon: int) -> Dict[str, Tensor]:
    first_mse=None; first_change=None; mse_steps=[]; delta_norm=torch.zeros(obs.state.shape[0], device=obs.state.device)
    with torch.no_grad():
        u0=system.encode(obs); base0=system.base_action(obs,u0); frozen_delta=system.adapter(u0, frozen_context, base0); frozen_executed=torch.clamp(base0+frozen_delta, -env.config.action_limit, env.config.action_limit)
        for _ in range(horizon):
            u=system.encode(obs); base=system.base_action(obs,u); delta=system.adapter(u, context, base); delta_norm=torch.linalg.norm(delta, dim=-1)
            executed=torch.clamp(base+delta, -env.config.action_limit, env.config.action_limit)
            expert=env.expert_action(obs, domain); mse=(executed-expert).square().mean(dim=-1)
            if first_mse is None:
                first_mse=mse; first_change=torch.linalg.norm(executed - frozen_executed, dim=-1)
            mse_steps.append(mse); obs=env.step(obs, domain, executed)
        return {"first_mse":first_mse,"first_action_change_vs_frozen":first_change,"query_mse":torch.stack(mse_steps).mean(dim=0),"final_error":env.goal_error(obs),"success":env.success(obs),"delta_norm":delta_norm}


def pack_rows(arrays: Dict[str, Tensor], *, meta: Dict[str, object], arm: str, s: int, u_passes: Optional[int], domain_ids: Tensor, query_repeats: int, z_source_domain: Optional[Tensor], latent_delta: Optional[Tensor], support_loss: Optional[Tensor]) -> List[Dict[str, object]]:
    rows=[]; num_domains=domain_ids.shape[0]//query_repeats
    cpu={k:v.detach().cpu() for k,v in arrays.items()}
    ids=domain_ids.detach().cpu(); src=None if z_source_domain is None else z_source_domain.detach().cpu(); ld=None if latent_delta is None else latent_delta.detach().cpu(); sl=None if support_loss is None else support_loss.detach().cpu()
    for domain_row in range(num_domains):
        start=domain_row*query_repeats; end=start+query_repeats
        source_idx=None if src is None else int(src[start])
        rows.append({**meta,"S":s,"U":u_passes,"support_mode":"common_base","domain_index":int(ids[start]),"query_indices":list(range(query_repeats)),"arm":arm,"z_source_domain_index":source_idx,"final_goal_errors":[float(x) for x in cpu["final_error"][start:end]],"successes":[bool(x) for x in cpu["success"][start:end]],"first_query_action_mse":[float(x) for x in cpu["first_mse"][start:end]],"first_action_change_vs_frozen":[float(x) for x in cpu["first_action_change_vs_frozen"][start:end]],"query_action_mse":[float(x) for x in cpu["query_mse"][start:end]],"delta_norm_last":[float(x) for x in cpu["delta_norm"][start:end]],"latent_delta_norm":None if ld is None or source_idx is None else float(ld[source_idx]),"support_loss":None if sl is None else float(sl[domain_row])})
    return rows


def per_domain_support_loss(system: FastAdapterSystem, records: Sequence[SupportRecord], z: Tensor, cfg: P16FastConfig) -> Tensor:
    return support_loss_per_domain(system, records, z, cfg)["total"].detach()


def calibration_records(system: nn.Module, env: Toy2DVectorEnv, domain: Toy2DDomain, *, seed: int, starts: int) -> List[SupportRecord]:
    records=[]
    for i in range(starts):
        records.extend(collect_common_base_records(system, env, domain, support_seed=seed+i, max_s=1))
    return records


def calibration_loss_rows(system: FastAdapterSystem, records: Sequence[SupportRecord], z_by_arm: Dict[str, Tensor], cfg: P16FastConfig, *, s: int, u_passes: int) -> List[Dict[str, object]]:
    rows=[]
    with torch.no_grad():
        for arm,z in z_by_arm.items():
            losses=[]
            for rec in records:
                losses.append(support_loss_per_domain(system, [rec], z, cfg)["mse"])
            stacked=torch.stack(losses).mean(dim=0).detach().cpu()
            for domain_index,value in enumerate(stacked):
                rows.append({"arm":arm,"S":s,"U":u_passes,"domain_index":domain_index,"prediction_mse":float(value),"prediction_units":"feature_u" if cfg.method=="feature_fast" else "delta_x"})
    return rows


def evaluate_fast(args: argparse.Namespace) -> Dict[str, object]:
    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    system, train_config, cfg=load_fast(Path(args.checkpoint), device=device)
    before=tensor_state_hash(system); toy=Toy2DConfig(); env=Toy2DVectorEnv(toy, device=device)
    domain=sample_toy2d_domains(args.num_domains, args.domain_seed, toy, device=device)
    query_domain=repeat_domain(domain, args.query_repeats)
    records=collect_common_base_records(system, env, domain, support_seed=args.support_seed, max_s=max(s for s, _u in args.fast_cells))
    cal=calibration_records(system, env, domain, seed=args.calibration_seed, starts=args.calibration_starts)
    perm=roll1_derangement(args.num_domains, device=device)
    rows=[]; support_rows=[]; cal_rows=[]
    for s,u_passes in args.fast_cells:
        with torch.enable_grad():
            z, metrics=adapt_fast(system, records, s=s, u_passes=u_passes, config=cfg, create_graph=False)
        z=z.detach(); frozen=system.z0.detach().unsqueeze(0).expand(args.num_domains,-1).clone()
        latent_delta=torch.linalg.norm(z - frozen, dim=-1)
        latent_zero=torch.zeros_like(latent_delta)
        support_loss_correct=per_domain_support_loss(system, records[:s], z, cfg) if s else torch.zeros(args.num_domains, device=device)
        support_loss_frozen=per_domain_support_loss(system, records[:s], frozen, cfg) if s else torch.zeros(args.num_domains, device=device)
        support_loss_swapped=per_domain_support_loss(system, records[:s], z[perm], cfg) if s else torch.zeros(args.num_domains, device=device)
        meta={"split":args.split,"train_seed":args.train_seed,"config_id":args.config_id,"domain_seed":args.domain_seed,"support_seed":args.support_seed,"query_seed":args.query_seed,"calibration_seed":args.calibration_seed,"query_repeats":args.query_repeats,"method":cfg.method}
        for m in metrics:
            support_rows.append({**meta,"S":s,"U":u_passes,**m})
        for cr in calibration_loss_rows(system, cal, {"frozen_z":frozen,"correct_z":z,"swapped_z":z[perm]}, cfg, s=s, u_passes=u_passes):
            cal_rows.append({**meta, **cr})
        z_correct=z.repeat_interleave(args.query_repeats, dim=0); z_frozen=frozen.repeat_interleave(args.query_repeats, dim=0); z_swapped=z[perm].repeat_interleave(args.query_repeats, dim=0)
        identity=torch.arange(args.num_domains, device=device).repeat_interleave(args.query_repeats); swapped_source=perm.repeat_interleave(args.query_repeats)
        arm_specs=[("base",z_correct,identity,None,None),("frozen_z",z_frozen,identity,latent_zero,support_loss_frozen),("correct_z",z_correct,identity,latent_delta,support_loss_correct),("swapped_z",z_swapped,swapped_source,latent_delta,support_loss_swapped)]
        for arm,z_arm,src,arm_latent,arm_support_loss in arm_specs:
            qobs=env.reset(args.num_domains*args.query_repeats, args.query_seed, domain=query_domain)
            arrays=rollout_fast_arrays(system, env, query_domain, qobs, z_arm, z_frozen, arm=arm, horizon=args.query_horizon)
            rows.extend(pack_rows(arrays, meta=meta, arm=arm, s=s, u_passes=u_passes, domain_ids=query_domain.domain_id, query_repeats=args.query_repeats, z_source_domain=src, latent_delta=arm_latent, support_loss=arm_support_loss))
    after=tensor_state_hash(system); outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"eval_raw.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    (outdir/"support_transition_metrics.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in support_rows))
    (outdir/"calibration_prediction_loss.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in cal_rows))
    summary={"raw_path":str(outdir/"eval_raw.jsonl"),"support_transition_metrics_path":str(outdir/"support_transition_metrics.jsonl"),"calibration_prediction_loss_path":str(outdir/"calibration_prediction_loss.jsonl"),"checkpoint":str(args.checkpoint),"checkpoint_sha256":sha256_file(args.checkpoint),"train_config":train_config,"freeze_audit":{"before":before,"after":after,"unchanged":before==after}}
    (outdir/"eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def evaluate_history(args: argparse.Namespace) -> Dict[str, object]:
    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    system, train_config, cfg=load_history(Path(args.checkpoint), device=device)
    before=tensor_state_hash(system); toy=Toy2DConfig(); env=Toy2DVectorEnv(toy, device=device)
    domain=sample_toy2d_domains(args.num_domains, args.domain_seed, toy, device=device)
    query_domain=repeat_domain(domain, args.query_repeats)
    records=collect_common_base_records(system, env, domain, support_seed=args.support_seed, max_s=max(args.history_s_values))
    rows=[]
    frozen_context=system.adapter.initial_context(args.num_domains, device=device)
    for s in args.history_s_values:
        context=history_update_context(system, records, s=s, batch_size=args.num_domains, device=device).detach()
        support_loss=None
        qcontext=context.repeat_interleave(args.query_repeats, dim=0); qfrozen=frozen_context.repeat_interleave(args.query_repeats, dim=0)
        qobs=env.reset(args.num_domains*args.query_repeats, args.query_seed, domain=query_domain)
        arrays=rollout_history_arrays(system, env, query_domain, qobs, qcontext, qfrozen, horizon=args.query_horizon)
        meta={"split":args.split,"train_seed":args.train_seed,"config_id":args.config_id,"domain_seed":args.domain_seed,"support_seed":args.support_seed,"query_seed":args.query_seed,"calibration_seed":args.calibration_seed,"query_repeats":args.query_repeats,"method":"history"}
        rows.extend(pack_rows(arrays, meta=meta, arm="history", s=s, u_passes=None, domain_ids=query_domain.domain_id, query_repeats=args.query_repeats, z_source_domain=None, latent_delta=None, support_loss=support_loss))
    after=tensor_state_hash(system); outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"eval_raw.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    summary={"raw_path":str(outdir/"eval_raw.jsonl"),"checkpoint":str(args.checkpoint),"checkpoint_sha256":sha256_file(args.checkpoint),"train_config":train_config,"freeze_audit":{"before":before,"after":after,"unchanged":before==after}}
    (outdir/"eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def ls_fit_delta(records: Sequence[SupportRecord]) -> Dict[str, Tensor]:
    # Per-domain, per-axis y = g*a + b using observable action and delta_x only.
    # Protocol requires double least-squares without ridge; rank deficient domains
    # are flagged by rank/condition and must fall back to base control.
    if not records:
        raise ValueError("least-squares diagnostic needs at least one support transition")
    actions = torch.stack([r.action.detach() for r in records], dim=1).double()  # [B,S,2]
    deltas = torch.stack([r.delta_x.detach() for r in records], dim=1).double()
    bsz = actions.shape[0]
    gains=[]; biases=[]; ranks=[]; conds=[]; singular_values=[]
    for axis in range(2):
        a = actions[..., axis]
        y = deltas[..., axis]
        X = torch.stack([a, torch.ones_like(a)], dim=-1)
        sv = torch.linalg.svdvals(X)
        tol = torch.finfo(torch.float64).eps * max(X.shape[1], X.shape[2]) * sv[:, :1]
        rank = (sv > tol).sum(dim=-1)
        cond = sv[:, 0] / torch.clamp(sv[:, -1], min=torch.finfo(torch.float64).tiny)
        solution = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)
        gains.append(solution[:,0]); biases.append(solution[:,1]); ranks.append(rank); conds.append(cond); singular_values.append(sv)
    return {"gain_hat":torch.stack(gains,dim=-1).float(),"bias_hat":torch.stack(biases,dim=-1).float(),"rank":torch.stack(ranks,dim=-1),"condition":torch.stack(conds,dim=-1).float(),"singular_values":torch.stack(singular_values,dim=1).float()}


def ls_predict(fit: Dict[str, Tensor], action: Tensor) -> tuple[Tensor, Tensor]:
    rank = fit["rank"]
    identifiable = (rank >= 2).all(dim=-1)
    pred = fit["gain_hat"].to(action.device) * action + fit["bias_hat"].to(action.device)
    return pred, identifiable.to(action.device)


def rollout_ls_arrays(env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, fit: Dict[str, Tensor], *, horizon: int) -> Dict[str, Tensor]:
    first_change = None
    saturation_steps = []
    with torch.no_grad():
        identifiable = (fit["rank"] >= 2).all(dim=-1).to(obs.state.device)
        for step in range(horizon):
            base = env.nominal_action(obs)
            g = torch.clamp(fit["gain_hat"].to(base.device), 0.1, 10.0)
            b = fit["bias_hat"].to(base.device)
            corrected = (base - b) / g
            residual = torch.clamp(corrected - base, -0.15, 0.15)
            action = torch.clamp(base + residual, -env.config.action_limit, env.config.action_limit)
            action = torch.where(identifiable.unsqueeze(-1), action, base)
            if first_change is None:
                first_change = torch.linalg.norm(action - base, dim=-1)
            saturation_steps.append((residual.abs() >= 0.15 - 1.0e-6).any(dim=-1).float() * identifiable.float())
            obs = env.step(obs, domain, action)
        return {"final_error": env.goal_error(obs), "success": env.success(obs), "first_action_change_vs_base": first_change, "saturation_fraction": torch.stack(saturation_steps).mean(dim=0), "identifiable": identifiable}


def evaluate_ls(args: argparse.Namespace) -> Dict[str, object]:
    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    toy=Toy2DConfig(); env=Toy2DVectorEnv(toy, device=device)
    system=build_feature_fast_system(P16FastConfig(config_id="ls_support", method="feature_fast"), device=device)
    for p in system.parameters(): p.requires_grad_(False)
    domain=sample_toy2d_domains(args.num_domains, args.domain_seed, toy, device=device)
    query_domain=repeat_domain(domain, args.query_repeats)
    records=collect_common_base_records(system, env, domain, support_seed=args.support_seed, max_s=max(args.s_values))
    cal=calibration_records(system, env, domain, seed=args.calibration_seed, starts=args.calibration_starts)
    rows=[]; diag=[]
    for s_val in args.s_values:
        fit=ls_fit_delta(records[:s_val])
        cal_mses=[]
        for rec in cal:
            pred, identifiable = ls_predict(fit, rec.action)
            mse=(pred - rec.delta_x).square().mean(dim=-1)
            cal_mses.append(torch.where(identifiable, mse, torch.full_like(mse, float('nan'))))
        cal_mse=torch.stack(cal_mses).nanmean(dim=0) if cal_mses else torch.full((args.num_domains,), float('nan'), device=device)
        identifiable=(fit["rank"] >= 2).all(dim=-1)
        qobs=env.reset(args.num_domains*args.query_repeats, args.query_seed, domain=query_domain)
        fit_rep={k:v.repeat_interleave(args.query_repeats, dim=0) for k,v in fit.items()}
        arrays=rollout_ls_arrays(env, query_domain, qobs, fit_rep, horizon=args.query_horizon)
        cpu={k:v.detach().cpu() for k,v in arrays.items()}
        ids=query_domain.domain_id.detach().cpu(); cal_cpu=cal_mse.detach().cpu(); rank=fit["rank"].detach().cpu(); cond=fit["condition"].detach().cpu(); sv=fit["singular_values"].detach().cpu(); ident=identifiable.detach().cpu()
        for d in range(args.num_domains):
            start=d*args.query_repeats; end=start+args.query_repeats
            rows.append({"split":args.split,"config_id":args.config_id,"train_seed":None,"method":"least_squares","arm":"least_squares","S":s_val,"U":None,"support_mode":"common_base","domain_seed":args.domain_seed,"support_seed":args.support_seed,"calibration_seed":args.calibration_seed,"query_seed":args.query_seed,"query_repeats":args.query_repeats,"domain_index":int(ids[start]),"query_indices":list(range(args.query_repeats)),"identifiable":bool(ident[d]),"rank":[int(x) for x in rank[d]],"condition":[float(x) for x in cond[d]],"singular_values":sv[d].tolist(),"heldout_displacement_mse":None if math.isnan(float(cal_cpu[d])) else float(cal_cpu[d]),"final_goal_errors":[float(x) for x in cpu["final_error"][start:end]],"successes":[bool(x) for x in cpu["success"][start:end]],"first_action_change_vs_base":[float(x) for x in cpu["first_action_change_vs_base"][start:end]],"saturation_fraction":[float(x) for x in cpu["saturation_fraction"][start:end]]})
        diag.append({"S":s_val,"identifiable_fraction":float(identifiable.float().mean().cpu()),"mean_heldout_displacement_mse":None if torch.isnan(cal_mse).all() else float(torch.nanmean(cal_mse).cpu())})
    outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    raw=outdir/"ls_raw.jsonl"; raw.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    summary={"raw_path":str(raw),"diagnostics":diag,"note":"Least-squares diagnostic uses only observable support action/delta_x. Rank-deficient domains are marked unidentifiable and use frozen base for control."}
    (outdir/"ls_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def write_plan(args: argparse.Namespace) -> Dict[str, object]:
    root=Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    plan={"status":"prepared_not_formally_run","scope":"P1.6 development-first diagnostic; no formal training/evaluation launched by write-plan","train_methods":[asdict(c) for c in fast_configs()]+[asdict(history_config())],"dev_train_seeds":TRAIN_SEEDS_DEV,"conditional_final_extra_seeds":TRAIN_SEEDS_FINAL_EXTRA,"fast_eval_cells":[{"S":s,"U":u} for s,u in FAST_CELLS],"history_eval_S":HISTORY_S_VALUES,"dev_seeds":{"domain_seed":410001,"support_seed":420001,"calibration_seed":430001,"query_seed":440001,"num_domains":128,"query_repeats":8},"optional_final_seeds":{"domain_seed":510001,"support_seed":520001,"calibration_seed":530001,"query_seed":540001,"num_domains":256,"query_repeats":8},"gate":"Before final, only displacement_fast eligible: correct_z vs same frozen_z at S3/U3 must improve SR in all three seeds, improve goal error in all three seeds, and mean SR gain >=2pp."}
    path=root/"P16_PREPARED_PLAN.json"; path.write_text(json.dumps(plan, indent=2, sort_keys=True)); return {"plan_path":str(path), **plan}


def counts(args: argparse.Namespace) -> Dict[str, object]:
    device=torch.device("cpu")
    result={"fast":{cfg.config_id: count_parameters(build_fast_system(cfg, device=device)) | asdict(cfg) for cfg in fast_configs()},"history":{history_config().config_id: count_parameters(build_history_system(history_config(), device=device)) | asdict(history_config())},"estimate_note":"Based on P1.5 local 4090: fast ~22-25s per 1500 updates; history ~16-18s. P16 displacement has same-sized predictor and extra support MSE, expect similar to feature_fast."}
    if args.output_dir:
        out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out/"P16_PARAM_COUNTS_AND_TIME_ESTIMATE.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_cells(items: Sequence[str]) -> List[tuple[int,int]]:
    cells=[]
    for item in items:
        if ":" in item:
            s,u=item.split(":",1)
        elif "," in item:
            s,u=item.split(",",1)
        else:
            raise argparse.ArgumentTypeError("fast cells must be S:U")
        cells.append((int(s), int(u)))
    return cells


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-id", required=True); p.add_argument("--train-seed", type=int, required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--updates", type=int, default=1500); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--support-s-train", type=int, default=3); p.add_argument("--gradient-u-train", type=int, default=3)
    p.add_argument("--query-horizon", type=int, default=10); p.add_argument("--train-stream-offset", type=int, default=25_000_000); p.add_argument("--device", default=None); p.add_argument("--log-every", type=int, default=500)


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", default=None); p.add_argument("--output-dir", required=True); p.add_argument("--config-id", required=True); p.add_argument("--train-seed", type=int, required=True); p.add_argument("--split", default="dev")
    p.add_argument("--domain-seed", type=int, required=True); p.add_argument("--support-seed", type=int, required=True); p.add_argument("--calibration-seed", type=int, required=True); p.add_argument("--query-seed", type=int, required=True)
    p.add_argument("--num-domains", type=int, required=True); p.add_argument("--query-repeats", type=int, default=8); p.add_argument("--query-horizon", type=int, default=10); p.add_argument("--calibration-starts", type=int, default=4); p.add_argument("--device", default=None)


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="P1.6 Fast Adapter diagnostic implementation."); sub=parser.add_subparsers(dest="cmd", required=True)
    p=sub.add_parser("train-fast"); add_train_args(p); p.add_argument("--method", choices=["feature_fast","displacement_fast"], required=True); p.set_defaults(func=train_fast)
    p=sub.add_parser("train-history"); add_train_args(p); p.set_defaults(func=train_history)
    p=sub.add_parser("eval-fast"); add_eval_args(p); p.add_argument("--fast-cells", nargs="+", default=[f"{s}:{u}" for s,u in FAST_CELLS]); p.set_defaults(func=lambda a: (setattr(a, 'fast_cells', parse_cells(a.fast_cells)) or evaluate_fast(a)))
    p=sub.add_parser("eval-history"); add_eval_args(p); p.add_argument("--history-s-values", type=int, nargs="+", default=HISTORY_S_VALUES); p.set_defaults(func=evaluate_history)
    p=sub.add_parser("eval-ls"); add_eval_args(p); p.add_argument("--s-values", type=int, nargs="+", default=[1,3,5]); p.set_defaults(func=evaluate_ls)
    p=sub.add_parser("write-plan"); p.add_argument("--output-dir", default=str(P16_ROOT)); p.set_defaults(func=write_plan)
    p=sub.add_parser("counts"); p.add_argument("--output-dir", default=None); p.set_defaults(func=counts)
    return parser.parse_args()


if __name__ == "__main__":
    ns=parse_args(); result=ns.func(ns); print(json.dumps(result, indent=2, sort_keys=True))
