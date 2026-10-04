from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

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
    inner_update_z,
    sample_toy2d_domains,
)

K_VALUES = [0, 1, 2, 3, 5, 10]
SUPPORT_MODES = ["common_base", "on_policy"]
TRAIN_SEEDS = [11, 22, 33, 44, 55]
Z_DIM_SWEEP = [4, 8, 16, 32, 64]
ALPHA_SWEEP = [0.05, 0.5, 1.0, 5.0]
HISTORY_WIDTHS = [256, 640]
P15_ROOT = Path("evaluations/fast_adapter_p15_v01")


@dataclass(frozen=True)
class FastConfig:
    config_id: str
    z_dim: int = 32
    alpha_mode: Literal["fixed", "learned"] = "fixed"
    inner_lr: float = 5.0
    updates: int = 1500
    batch_size: int = 64
    support_k_train: int = 3
    query_horizon: int = 10
    lr: float = 1.0e-3
    lambda_cos: float = 0.1
    lambda_z: float = 1.0e-4
    residual_reg: float = 1.0e-3
    train_stream_offset: int = 1500000


@dataclass(frozen=True)
class HistoryConfig:
    config_id: str
    width: int
    updates: int = 1500
    batch_size: int = 64
    support_k_train: int = 3
    query_horizon: int = 10
    lr: float = 1.0e-3
    residual_reg: float = 1.0e-3
    train_stream_offset: int = 1500000


@dataclass(frozen=True)
class TransitionBatch:
    u: Tensor
    action: Tensor
    u_next: Tensor


def alpha_token(alpha: float) -> str:
    return ("%g" % alpha).replace(".", "p")


def fast_grid() -> List[FastConfig]:
    candidates: List[tuple[int, str, float]] = []
    candidates.extend((z_dim, "fixed", 5.0) for z_dim in Z_DIM_SWEEP)
    candidates.extend((32, "fixed", alpha) for alpha in ALPHA_SWEEP)
    candidates.append((32, "learned", 0.5))
    dedup: Dict[tuple[int, str, float], FastConfig] = {}
    for z_dim, alpha_mode, inner_lr in candidates:
        key = (z_dim, alpha_mode, float(inner_lr))
        if key in dedup:
            continue
        if alpha_mode == "learned":
            config_id = f"z{z_dim}_alpha_learned"
        else:
            config_id = f"z{z_dim}_alpha{alpha_token(inner_lr)}"
        dedup[key] = FastConfig(config_id=config_id, z_dim=z_dim, alpha_mode=alpha_mode, inner_lr=float(inner_lr))
    return list(dedup.values())


def history_grid() -> List[HistoryConfig]:
    return [HistoryConfig(config_id=f"history_width{width}", width=width) for width in HISTORY_WIDTHS]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def count_parameters(module: nn.Module) -> Dict[str, int]:
    return {"total": sum(p.numel() for p in module.parameters()), "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}


def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().square().sum().cpu())
    return math.sqrt(total)


def build_fast_system(config: FastConfig, *, device: torch.device) -> FastAdapterSystem:
    toy = Toy2DConfig()
    system = FastAdapterSystem(
        encoder=InteractionEncoder(obj_dim=2, tar_dim=2, proprio_dim=2, action_dim=2),
        base_policy=ScriptedNominalPolicy2D(action_limit=toy.action_limit),
        adapter=FastAdapter(z_dim=config.z_dim, max_residual=torch.tensor([0.15, 0.15])),
        dynamics=DynamicsPredictor(z_dim=config.z_dim),
        z_dim=config.z_dim,
        learn_inner_lr=config.alpha_mode == "learned",
        inner_lr=config.inner_lr,
    )
    return system.to(device)


class P15HistoryAdapter(nn.Module):
    def __init__(self, *, u_dim: int = 256, action_dim: int = 2, context_dim: int = 32, width: int = 256, max_residual: float = 0.15):
        super().__init__()
        self.context_dim = context_dim
        self.register_buffer("max_residual", torch.full((action_dim,), float(max_residual)))
        self.transition_proj = nn.Sequential(nn.Linear(u_dim + action_dim + u_dim, 128), nn.GELU(), nn.Linear(128, context_dim))
        self.rnn = nn.GRUCell(context_dim, context_dim)
        self.policy = nn.Sequential(nn.Linear(u_dim + context_dim + action_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, action_dim))

    def initial_context(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.context_dim, device=device)

    def update_context(self, context: Tensor, u: Tensor, action: Tensor, u_next: Tensor) -> Tensor:
        transition = torch.cat([u, action.detach(), u_next - u], dim=-1)
        return self.rnn(self.transition_proj(transition), context)

    def forward(self, u: Tensor, context: Tensor, a_base: Tensor) -> Tensor:
        raw = self.policy(torch.cat([u, context, a_base], dim=-1))
        return self.max_residual.to(raw.device) * torch.tanh(raw)


class P15HistorySystem(nn.Module):
    def __init__(self, config: HistoryConfig):
        super().__init__()
        toy = Toy2DConfig()
        self.encoder = InteractionEncoder(obj_dim=2, tar_dim=2, proprio_dim=2, action_dim=2)
        self.base_policy = ScriptedNominalPolicy2D(action_limit=toy.action_limit)
        self.adapter = P15HistoryAdapter(width=config.width)
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()

    def encode(self, obs: Toy2DObservation) -> Tensor:
        runtime = obs.adapter_observation()
        return self.encoder(runtime.obj_feat, runtime.tar_feat, runtime.proprio, runtime.prev_action)

    def base_action(self, obs: Toy2DObservation, u: Tensor) -> Tensor:
        with torch.no_grad():
            return self.base_policy(obs.adapter_observation(), u)


def repeat_domain(domain: Toy2DDomain, repeats: int) -> Toy2DDomain:
    return Toy2DDomain(gain=domain.gain.repeat_interleave(repeats, dim=0), bias=domain.bias.repeat_interleave(repeats, dim=0), domain_id=domain.domain_id.repeat_interleave(repeats, dim=0))


def roll1_derangement(n: int, *, device: torch.device) -> Tensor:
    if n < 2:
        raise ValueError("roll1 swapped-z diagnostic requires at least two domains")
    return torch.roll(torch.arange(n, device=device), shifts=1)


def adaptation_terms_per_sample(u_hat_next: Tensor, u_next: Tensor, z: Tensor, z0: Tensor, loss_config: AdaptationLossConfig) -> Dict[str, Tensor]:
    target = u_next.detach()
    mse = (u_hat_next - target).square().mean(dim=-1)
    cos = 1.0 - F.cosine_similarity(u_hat_next, target, dim=-1)
    z_ref = z0.detach().unsqueeze(0).expand_as(z)
    z_reg = (z - z_ref).square().mean(dim=-1)
    adapt = mse + loss_config.lambda_cos * cos + loss_config.lambda_z * z_reg
    return {"mse": mse, "cos": cos, "z_reg": z_reg, "adapt": adapt}


def terms_mean(terms: Dict[str, Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().mean().cpu()) for key, value in terms.items()}


def update_z_from_transition(system: FastAdapterSystem, transition: TransitionBatch, z: Tensor, loss_config: AdaptationLossConfig, *, create_graph: bool) -> tuple[Tensor, Dict[str, float]]:
    pre = adaptation_terms_per_sample(system.dynamics(transition.u, transition.action, z), transition.u_next, z, system.z0, loss_config)
    update = inner_update_z(system=system, u=transition.u, action_observed=transition.action, z=z, u_next=transition.u_next, loss_config=loss_config, create_graph=create_graph, detach_executed_action=True)
    post = adaptation_terms_per_sample(system.dynamics(transition.u, transition.action, update.z_next), transition.u_next, update.z_next, system.z0, loss_config)
    metrics = {f"pre_{k}": v for k, v in terms_mean(pre).items()}
    metrics.update({f"post_{k}": v for k, v in terms_mean(post).items()})
    metrics["grad_z_norm"] = float(torch.linalg.norm(update.grad_z.detach(), dim=-1).mean().cpu())
    return update.z_next, metrics


def support_transition_from_action(system: FastAdapterSystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, action: Tensor) -> tuple[Toy2DObservation, TransitionBatch]:
    executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
    u = system.encode(obs.adapter_observation())
    next_obs = env.step(obs, domain, executed.detach())
    u_next = system.encode(next_obs.adapter_observation())
    return next_obs, TransitionBatch(u=u, action=next_obs.prev_action.detach(), u_next=u_next)


def support_adapt_fast_on_policy(system: FastAdapterSystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, *, k: int, loss_config: AdaptationLossConfig, create_graph: bool) -> tuple[Tensor, List[Dict[str, float]]]:
    z = expand_latent(system.z0, domain.gain.shape[0])
    metrics: List[Dict[str, float]] = []
    for support_index in range(k):
        action, _delta, _base, _u = system.act(obs.adapter_observation(), z)
        obs, transition = support_transition_from_action(system, env, domain, obs, action)
        z, step_metrics = update_z_from_transition(system, transition, z, loss_config, create_graph=create_graph)
        step_metrics.update({"support_index": support_index, "support_mode": "on_policy"})
        metrics.append(step_metrics)
    return z, metrics


def collect_common_base_transitions(system: FastAdapterSystem, env: Toy2DVectorEnv, domain: Toy2DDomain, *, support_seed: int, max_k: int) -> List[TransitionBatch]:
    obs = env.reset(domain.gain.shape[0], support_seed, domain=domain)
    transitions: List[TransitionBatch] = []
    for _ in range(max_k):
        u = system.encode(obs.adapter_observation())
        base = system.base_action(obs.adapter_observation(), u)
        obs, transition = support_transition_from_action(system, env, domain, obs, base)
        transitions.append(transition)
    return transitions


def support_adapt_fast_common_base(system: FastAdapterSystem, transitions: List[TransitionBatch], *, k: int, loss_config: AdaptationLossConfig, create_graph: bool) -> tuple[Tensor, List[Dict[str, float]]]:
    z = expand_latent(system.z0, transitions[0].u.shape[0])
    metrics: List[Dict[str, float]] = []
    for support_index, transition in enumerate(transitions[:k]):
        z, step_metrics = update_z_from_transition(system, transition, z, loss_config, create_graph=create_graph)
        step_metrics.update({"support_index": support_index, "support_mode": "common_base"})
        metrics.append(step_metrics)
    return z, metrics


def collect_base_calibration_transitions(system: FastAdapterSystem, env: Toy2DVectorEnv, domain: Toy2DDomain, *, seed: int, starts: int = 4) -> List[TransitionBatch]:
    transitions: List[TransitionBatch] = []
    for start_index in range(starts):
        obs = env.reset(domain.gain.shape[0], seed + start_index, domain=domain)
        u = system.encode(obs.adapter_observation())
        base = system.base_action(obs.adapter_observation(), u)
        _next_obs, transition = support_transition_from_action(system, env, domain, obs, base)
        transitions.append(transition)
    return transitions


def validation_adapt_loss(system: FastAdapterSystem, transitions: List[TransitionBatch], z: Tensor, loss_config: AdaptationLossConfig) -> Dict[str, float]:
    rows = []
    with torch.no_grad():
        for transition in transitions:
            terms = adaptation_terms_per_sample(system.dynamics(transition.u, transition.action, z.detach()), transition.u_next, z.detach(), system.z0.detach(), loss_config)
            rows.append(terms_mean(terms))
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


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
    return loss, {"first_query_action_mse": float(first_mse), "query_action_mse": float(torch.stack([item.detach() for item in action_losses]).mean().cpu()), "final_goal_error": float(final_error.mean().cpu()), "success_rate": float(success.float().mean().cpu())}


def train_fast(args: argparse.Namespace) -> Dict[str, object]:
    cfg = FastConfig(config_id=args.config_id, z_dim=args.z_dim, alpha_mode=args.alpha_mode, inner_lr=args.inner_lr, updates=args.updates, batch_size=args.batch_size, support_k_train=args.support_k_train, query_horizon=args.query_horizon, train_stream_offset=args.train_stream_offset)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.train_seed)
    toy = Toy2DConfig()
    env = Toy2DVectorEnv(toy, device=device)
    system = build_fast_system(cfg, device=device)
    opt = torch.optim.Adam([p for p in system.parameters() if p.requires_grad], lr=cfg.lr)
    loss_cfg = AdaptationLossConfig(lambda_cos=cfg.lambda_cos, lambda_z=cfg.lambda_z)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = {"mode": "p15_fast", "train_seed": args.train_seed, "fast_config": asdict(cfg), "toy_config": toy.to_dict(), "param_count": count_parameters(system)}
    (outdir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    history_path = outdir / "history.jsonl"
    started = time.perf_counter()
    records = []
    with history_path.open("w") as handle:
        for update_idx in range(cfg.updates):
            step_start = time.perf_counter()
            stream = cfg.train_stream_offset + args.train_seed * 100000 + update_idx
            domain = sample_toy2d_domains(cfg.batch_size, stream, toy, device=device)
            support_obs = env.reset(cfg.batch_size, stream + 10000000, domain=domain)
            z, support_metrics = support_adapt_fast_on_policy(system, env, domain, support_obs, k=cfg.support_k_train, loss_config=loss_cfg, create_graph=True)
            query_obs = env.reset(cfg.batch_size, stream + 20000000, domain=domain)
            loss, query_metrics = query_loss_fast(system, env, domain, query_obs, z, horizon=cfg.query_horizon, residual_reg=cfg.residual_reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = {"encoder": grad_norm(system.encoder.parameters()), "adapter": grad_norm(system.adapter.parameters()), "dynamics": grad_norm(system.dynamics.parameters()), "z0": grad_norm([system.z0]), "log_inner_lr": grad_norm([system.log_inner_lr]) if system.log_inner_lr is not None else 0.0}
            opt.step()
            finite = bool(torch.isfinite(loss.detach()).item() and torch.isfinite(system.inner_lr.detach()).item())
            rec = {"update": update_idx + 1, "loss": float(loss.detach().cpu()), "finite": finite, "inner_lr": float(system.inner_lr.detach().cpu()), "latent_delta_norm": float(torch.linalg.norm(z.detach() - system.z0.detach().unsqueeze(0), dim=-1).mean().cpu()), "sec_update": time.perf_counter() - step_start, "support_transition_metrics": support_metrics, **query_metrics, "grad_norms": grads}
            records.append(rec)
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            if not finite:
                break
            if args.log_every and (update_idx == 0 or (update_idx + 1) % args.log_every == 0):
                print(json.dumps(rec, sort_keys=True))
    elapsed = time.perf_counter() - started
    ckpt_path = outdir / "checkpoint.pt"
    torch.save({"system": system.state_dict(), "config": config, "history_tail": records[-10:]}, ckpt_path)
    final_parameter_allfinite = all(torch.isfinite(p.detach()).all().item() for p in system.parameters())
    summary = {"checkpoint": str(ckpt_path), "history": str(history_path), "elapsed_sec": elapsed, "mean_sec_update": elapsed / max(1, len(records)), "completed_updates": len(records), "final_parameter_allfinite": final_parameter_allfinite, "last": records[-1]}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def support_context_history_on_policy(system: P15HistorySystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, *, k: int) -> Tensor:
    context = system.adapter.initial_context(domain.gain.shape[0], device=domain.gain.device)
    for _ in range(k):
        u = system.encode(obs)
        base = system.base_action(obs, u)
        delta = system.adapter(u, context, base)
        executed = torch.clamp(base + delta, -env.config.action_limit, env.config.action_limit)
        next_obs = env.step(obs, domain, executed.detach())
        u_next = system.encode(next_obs)
        context = system.adapter.update_context(context, u, next_obs.prev_action, u_next)
        obs = next_obs
    return context


def support_context_history_common_base(system: P15HistorySystem, transitions: List[TransitionBatch], *, batch_size: int, device: torch.device, k: int) -> Tensor:
    context = system.adapter.initial_context(batch_size, device=device)
    for transition in transitions[:k]:
        context = system.adapter.update_context(context, transition.u, transition.action, transition.u_next)
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
    return loss, {"first_query_action_mse": float(first_mse), "query_action_mse": float(torch.stack([item.detach() for item in action_losses]).mean().cpu()), "final_goal_error": float(final_error.mean().cpu()), "success_rate": float(success.float().mean().cpu())}


def train_history(args: argparse.Namespace) -> Dict[str, object]:
    cfg = HistoryConfig(config_id=args.config_id, width=args.width, updates=args.updates, batch_size=args.batch_size, support_k_train=args.support_k_train, query_horizon=args.query_horizon, train_stream_offset=args.train_stream_offset)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.train_seed)
    toy = Toy2DConfig()
    env = Toy2DVectorEnv(toy, device=device)
    system = P15HistorySystem(cfg).to(device)
    opt = torch.optim.Adam([p for p in system.parameters() if p.requires_grad], lr=cfg.lr)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = {"mode": "p15_history", "train_seed": args.train_seed, "history_config": asdict(cfg), "toy_config": toy.to_dict(), "param_count": count_parameters(system)}
    (outdir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    history_path = outdir / "history.jsonl"
    records = []
    started = time.perf_counter()
    with history_path.open("w") as handle:
        for update_idx in range(cfg.updates):
            step_start = time.perf_counter()
            stream = cfg.train_stream_offset + args.train_seed * 100000 + update_idx
            domain = sample_toy2d_domains(cfg.batch_size, stream, toy, device=device)
            support_obs = env.reset(cfg.batch_size, stream + 10000000, domain=domain)
            context = support_context_history_on_policy(system, env, domain, support_obs, k=cfg.support_k_train)
            query_obs = env.reset(cfg.batch_size, stream + 20000000, domain=domain)
            loss, query_metrics = query_loss_history(system, env, domain, query_obs, context, horizon=cfg.query_horizon, residual_reg=cfg.residual_reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = {"encoder": grad_norm(system.encoder.parameters()), "adapter": grad_norm(system.adapter.parameters())}
            opt.step()
            finite = bool(torch.isfinite(loss.detach()).item())
            rec = {"update": update_idx + 1, "loss": float(loss.detach().cpu()), "finite": finite, "sec_update": time.perf_counter() - step_start, **query_metrics, "grad_norms": grads}
            records.append(rec)
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            if not finite:
                break
            if args.log_every and (update_idx == 0 or (update_idx + 1) % args.log_every == 0):
                print(json.dumps(rec, sort_keys=True))
    elapsed = time.perf_counter() - started
    ckpt_path = outdir / "checkpoint.pt"
    torch.save({"system": system.state_dict(), "config": config, "history_tail": records[-10:]}, ckpt_path)
    final_parameter_allfinite = all(torch.isfinite(p.detach()).all().item() for p in system.parameters())
    summary = {"checkpoint": str(ckpt_path), "history": str(history_path), "elapsed_sec": elapsed, "mean_sec_update": elapsed / max(1, len(records)), "completed_updates": len(records), "final_parameter_allfinite": final_parameter_allfinite, "last": records[-1]}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_fast_system(checkpoint: Path, *, device: torch.device) -> tuple[FastAdapterSystem, Dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = FastConfig(**ckpt["config"]["fast_config"])
    system = build_fast_system(cfg, device=device)
    system.load_state_dict(ckpt["system"])
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    return system, ckpt["config"]


def pack_domain_rows(per_query: Dict[str, Tensor], *, meta: Dict[str, object], arm: str, k: int, support_mode: str, domain_ids: Tensor, query_repeats: int, z_source_domain: Tensor, latent_delta: Optional[Tensor]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    num_domains = domain_ids.shape[0] // query_repeats
    for domain_row in range(num_domains):
        start = domain_row * query_repeats
        end = start + query_repeats
        domain_index = int(domain_ids[start].detach().cpu())
        source_idx = None if z_source_domain is None else int(z_source_domain[start].detach().cpu())
        rows.append({
            **meta,
            "support_mode": support_mode,
            "domain_index": domain_index,
            "query_indices": list(range(query_repeats)),
            "arm": arm,
            "K": k,
            "z_source_domain_index": source_idx,
            "final_goal_errors": [float(x) for x in per_query["final_error"][start:end].detach().cpu()],
            "successes": [bool(x) for x in per_query["success"][start:end].detach().cpu()],
            "first_query_action_mse": [float(x) for x in per_query["first_mse"][start:end].detach().cpu()],
            "query_action_mse": [float(x) for x in per_query["query_mse"][start:end].detach().cpu()],
            "delta_norm_last": [float(x) for x in per_query["delta_norm"][start:end].detach().cpu()],
            "latent_delta_norm": None if latent_delta is None or source_idx is None else float(latent_delta[source_idx].detach().cpu()),
        })
    return rows


def rollout_query_fast_arrays(system: FastAdapterSystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, z: Tensor, *, arm: str, horizon: int) -> Dict[str, Tensor]:
    first_mse = None
    mse_steps = []
    delta_norm = torch.zeros(obs.state.shape[0], device=obs.state.device)
    with torch.no_grad():
        for _ in range(horizon):
            if arm == "base":
                action = env.nominal_action(obs)
                delta_norm = torch.zeros(action.shape[0], device=action.device)
            else:
                action, delta, _base, _u = system.act(obs.adapter_observation(), z)
                delta_norm = torch.linalg.norm(delta, dim=-1)
            expert = env.expert_action(obs, domain)
            executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
            mse = (executed - expert).square().mean(dim=-1)
            if first_mse is None:
                first_mse = mse
            mse_steps.append(mse)
            obs = env.step(obs, domain, executed)
        final_error = env.goal_error(obs)
        success = env.success(obs)
        query_mse = torch.stack(mse_steps).mean(dim=0)
    return {"first_mse": first_mse, "query_mse": query_mse, "final_error": final_error, "success": success, "delta_norm": delta_norm}


def evaluate_fast(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint)
    system, train_config = load_fast_system(checkpoint, device=device)
    before_hash = tensor_state_hash(system)
    toy = Toy2DConfig()
    env = Toy2DVectorEnv(toy, device=device)
    domain = sample_toy2d_domains(args.num_domains, args.domain_seed, toy, device=device)
    query_domain = repeat_domain(domain, args.query_repeats)
    loss_cfg = AdaptationLossConfig(lambda_cos=args.lambda_cos, lambda_z=args.lambda_z)
    calibration = collect_base_calibration_transitions(system, env, domain, seed=args.calibration_seed, starts=args.calibration_starts)
    common_base_transitions = collect_common_base_transitions(system, env, domain, support_seed=args.support_seed, max_k=max(args.k_values))
    perm = roll1_derangement(args.num_domains, device=device)
    rows: List[Dict[str, object]] = []
    support_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    for support_mode in args.support_modes:
        for k in args.k_values:
            if support_mode == "common_base":
                with torch.enable_grad():
                    z, support_metrics = support_adapt_fast_common_base(system, common_base_transitions, k=k, loss_config=loss_cfg, create_graph=False)
            else:
                support_obs = env.reset(args.num_domains, args.support_seed, domain=domain)
                with torch.enable_grad():
                    z, support_metrics = support_adapt_fast_on_policy(system, env, domain, support_obs, k=k, loss_config=loss_cfg, create_graph=False)
            z = z.detach()
            latent_delta = torch.linalg.norm(z - system.z0.detach().unsqueeze(0), dim=-1)
            frozen = system.z0.detach().unsqueeze(0).expand(args.num_domains, -1).clone()
            for arm_name, z_cal in [("correct_z", z), ("frozen_z", frozen)]:
                cal = validation_adapt_loss(system, calibration, z_cal, loss_cfg)
                calibration_rows.append({"support_mode": support_mode, "arm": arm_name, "K": k, **cal})
            for item in support_metrics:
                support_rows.append({"K": k, **item})
            z_correct = z.repeat_interleave(args.query_repeats, dim=0)
            z_frozen = frozen.repeat_interleave(args.query_repeats, dim=0)
            z_swapped = z[perm].repeat_interleave(args.query_repeats, dim=0)
            identity = torch.arange(args.num_domains, device=device).repeat_interleave(args.query_repeats)
            swapped_source = perm.repeat_interleave(args.query_repeats)
            meta = {"split": args.split, "train_seed": args.train_seed, "config_id": args.config_id, "domain_seed": args.domain_seed, "support_seed": args.support_seed, "query_seed": args.query_seed, "calibration_seed": args.calibration_seed, "query_repeats": args.query_repeats}
            for arm, z_arm, source in [("base", z_correct, identity), ("frozen_z", z_frozen, identity), ("correct_z", z_correct, identity), ("swapped_z", z_swapped, swapped_source)]:
                query_obs = env.reset(args.num_domains * args.query_repeats, args.query_seed, domain=query_domain)
                arrays = rollout_query_fast_arrays(system, env, query_domain, query_obs, z_arm, arm=arm, horizon=args.query_horizon)
                rows.extend(pack_domain_rows(arrays, meta=meta, arm=arm, k=k, support_mode=support_mode, domain_ids=query_domain.domain_id, query_repeats=args.query_repeats, z_source_domain=source, latent_delta=latent_delta))
    after_hash = tensor_state_hash(system)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "eval_raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    support_path = outdir / "support_transition_metrics.jsonl"
    support_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in support_rows))
    cal_path = outdir / "calibration_adapt_loss.jsonl"
    cal_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in calibration_rows))
    summary = {"raw_path": str(raw_path), "support_transition_metrics_path": str(support_path), "calibration_adapt_loss_path": str(cal_path), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "train_config": train_config, "freeze_audit": {"before": before_hash, "after": after_hash, "unchanged": before_hash == after_hash}}
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_history_system(checkpoint: Path, *, device: torch.device) -> tuple[P15HistorySystem, Dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = HistoryConfig(**ckpt["config"]["history_config"])
    system = P15HistorySystem(cfg).to(device)
    system.load_state_dict(ckpt["system"])
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    return system, ckpt["config"]


def collect_common_base_transitions_history(system: P15HistorySystem, env: Toy2DVectorEnv, domain: Toy2DDomain, *, support_seed: int, max_k: int) -> List[TransitionBatch]:
    obs = env.reset(domain.gain.shape[0], support_seed, domain=domain)
    transitions: List[TransitionBatch] = []
    for _ in range(max_k):
        u = system.encode(obs)
        base = system.base_action(obs, u)
        executed = torch.clamp(base, -env.config.action_limit, env.config.action_limit)
        next_obs = env.step(obs, domain, executed.detach())
        u_next = system.encode(next_obs)
        transitions.append(TransitionBatch(u=u, action=next_obs.prev_action.detach(), u_next=u_next))
        obs = next_obs
    return transitions


def rollout_query_history_arrays(system: P15HistorySystem, env: Toy2DVectorEnv, domain: Toy2DDomain, obs: Toy2DObservation, context: Tensor, *, horizon: int) -> Dict[str, Tensor]:
    first_mse = None
    mse_steps = []
    delta_norm = torch.zeros(obs.state.shape[0], device=obs.state.device)
    with torch.no_grad():
        for _ in range(horizon):
            u = system.encode(obs)
            base = system.base_action(obs, u)
            delta = system.adapter(u, context, base)
            delta_norm = torch.linalg.norm(delta, dim=-1)
            executed = torch.clamp(base + delta, -env.config.action_limit, env.config.action_limit)
            expert = env.expert_action(obs, domain)
            mse = (executed - expert).square().mean(dim=-1)
            if first_mse is None:
                first_mse = mse
            mse_steps.append(mse)
            obs = env.step(obs, domain, executed)
        final_error = env.goal_error(obs)
        success = env.success(obs)
        query_mse = torch.stack(mse_steps).mean(dim=0)
    return {"first_mse": first_mse, "query_mse": query_mse, "final_error": final_error, "success": success, "delta_norm": delta_norm}


def evaluate_history(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint)
    system, train_config = load_history_system(checkpoint, device=device)
    before_hash = tensor_state_hash(system)
    toy = Toy2DConfig()
    env = Toy2DVectorEnv(toy, device=device)
    domain = sample_toy2d_domains(args.num_domains, args.domain_seed, toy, device=device)
    query_domain = repeat_domain(domain, args.query_repeats)
    common_base_transitions = collect_common_base_transitions_history(system, env, domain, support_seed=args.support_seed, max_k=max(args.k_values))
    rows: List[Dict[str, object]] = []
    for support_mode in args.support_modes:
        for k in args.k_values:
            if support_mode == "common_base":
                context = support_context_history_common_base(system, common_base_transitions, batch_size=args.num_domains, device=device, k=k)
            else:
                support_obs = env.reset(args.num_domains, args.support_seed, domain=domain)
                context = support_context_history_on_policy(system, env, domain, support_obs, k=k)
            query_context = context.detach().repeat_interleave(args.query_repeats, dim=0)
            query_obs = env.reset(args.num_domains * args.query_repeats, args.query_seed, domain=query_domain)
            arrays = rollout_query_history_arrays(system, env, query_domain, query_obs, query_context, horizon=args.query_horizon)
            meta = {"split": args.split, "train_seed": args.train_seed, "config_id": args.config_id, "domain_seed": args.domain_seed, "support_seed": args.support_seed, "query_seed": args.query_seed, "calibration_seed": args.calibration_seed, "query_repeats": args.query_repeats}
            packed = pack_domain_rows(arrays, meta=meta, arm="history", k=k, support_mode=support_mode, domain_ids=query_domain.domain_id, query_repeats=args.query_repeats, z_source_domain=None, latent_delta=None)
            for row in packed:
                row["support_loss"] = None
            rows.extend(packed)
    after_hash = tensor_state_hash(system)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "eval_raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {"raw_path": str(raw_path), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "train_config": train_config, "freeze_audit": {"before": before_hash, "after": after_hash, "unchanged": before_hash == after_hash}}
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def write_plan(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "prepared_not_formally_run",
        "fast_configs": [asdict(cfg) for cfg in fast_grid()],
        "history_configs": [asdict(cfg) for cfg in history_grid()],
        "train_seeds": TRAIN_SEEDS,
        "K": K_VALUES,
        "support_modes": SUPPORT_MODES,
        "dev_seeds": {"domain_seed": 131000, "support_seed": 132000, "calibration_seed": 133000, "query_seed": 141000, "num_domains": 128, "query_repeats": 8},
        "final_seeds": {"domain_seed": 191000, "support_seed": 192000, "calibration_seed": 193000, "query_seed": 201000, "num_domains": 256, "query_repeats": 8},
        "selection": {"fast": "dev common_base K3 mean goal error averaged across five train seeds; tie 1e-5 prefers smaller dim, fixed alpha, smaller alpha", "history": "dev common_base K3 mean goal error averaged across five train seeds; tie prefers width256", "strongest_fixed_background": "dev K0"},
        "counts": {"fast_train_runs": len(fast_grid()) * len(TRAIN_SEEDS), "history_train_runs": len(history_grid()) * len(TRAIN_SEEDS), "total_train_runs": len(fast_grid()) * len(TRAIN_SEEDS) + len(history_grid()) * len(TRAIN_SEEDS)},
        "scope": "P1.5 preparation only; no official training/evaluation has been launched by write-plan.",
    }
    path = root / "P15_PREPARED_PLAN.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return {"plan_path": str(path), **plan}


def counts(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device("cpu")
    fast_counts = {cfg.config_id: count_parameters(build_fast_system(cfg, device=device)) | {"z_dim": cfg.z_dim, "alpha_mode": cfg.alpha_mode, "inner_lr": cfg.inner_lr} for cfg in fast_grid()}
    history_counts = {cfg.config_id: count_parameters(P15HistorySystem(cfg)) | {"width": cfg.width} for cfg in history_grid()}
    estimates = {"p1_observed_fast_sec_per_1500": 19.0, "p1_observed_history256_sec_per_1500": 15.5, "fast_45_runs_minutes_at_p1_rate": 45 * 19.0 / 60.0, "history_10_runs_minutes_at_p1_rate": 10 * 15.5 / 60.0, "total_minutes_at_p1_rate_before_eval": (45 * 19.0 + 10 * 15.5) / 60.0, "note": "Actual P1.5 varies with z_dim/history width; this is a local 4090 planning estimate, not a run result."}
    result = {"fast": fast_counts, "history": history_counts, "estimates": estimates}
    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "P15_PARAM_COUNTS_AND_TIME_ESTIMATE.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-k-train", type=int, default=3)
    parser.add_argument("--query-horizon", type=int, default=10)
    parser.add_argument("--train-stream-offset", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=250)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--domain-seed", type=int, required=True)
    parser.add_argument("--support-seed", type=int, required=True)
    parser.add_argument("--query-seed", type=int, required=True)
    parser.add_argument("--calibration-seed", type=int, required=True)
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--query-repeats", type=int, default=8)
    parser.add_argument("--query-horizon", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--support-modes", nargs="+", choices=SUPPORT_MODES, default=SUPPORT_MODES)
    parser.add_argument("--device", default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1.5 Fast Adapter ablation preparation/training/eval entrypoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train-fast")
    add_common_train_args(p)
    p.add_argument("--z-dim", type=int, required=True)
    p.add_argument("--alpha-mode", choices=["fixed", "learned"], required=True)
    p.add_argument("--inner-lr", type=float, required=True)
    p.set_defaults(func=train_fast)
    p = sub.add_parser("train-history")
    add_common_train_args(p)
    p.add_argument("--width", type=int, required=True)
    p.set_defaults(func=train_history)
    p = sub.add_parser("eval-fast")
    add_eval_args(p)
    p.add_argument("--lambda-cos", type=float, default=0.1)
    p.add_argument("--lambda-z", type=float, default=1.0e-4)
    p.add_argument("--calibration-starts", type=int, default=4)
    p.set_defaults(func=evaluate_fast)
    p = sub.add_parser("eval-history")
    add_eval_args(p)
    p.set_defaults(func=evaluate_history)
    p = sub.add_parser("write-plan")
    p.add_argument("--output-dir", default=str(P15_ROOT))
    p.set_defaults(func=write_plan)
    p = sub.add_parser("counts")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=counts)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if hasattr(parsed, "train_stream_offset") and parsed.train_stream_offset is None:
        parsed.train_stream_offset = 1500000
    print(json.dumps(parsed.func(parsed), indent=2, sort_keys=True))
