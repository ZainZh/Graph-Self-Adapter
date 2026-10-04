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

from prove.fast_adapter import Toy2DConfig, Toy2DDomain, Toy2DObservation, Toy2DVectorEnv, sample_toy2d_domains
from tools.fast_adapter.p15_experiment import (
    P15HistorySystem,
    count_parameters,
    grad_norm,
    repeat_domain,
    roll1_derangement,
    sha256_file,
    tensor_state_hash,
)
from tools.fast_adapter.p16_experiment import SupportRecord, collect_common_base_records, history_update_context, ls_fit_delta

P17_ROOT = Path("evaluations/fast_adapter_p17_v01")
P16_HISTORY_ROOT = Path("evaluations/fast_adapter_p16_v01/train/history")
P16_HISTORY_CHECKPOINTS = {seed: P16_HISTORY_ROOT / f"seed{seed}" / "checkpoint.pt" for seed in [11, 22, 33, 44, 55]}
TRAIN_SEEDS_DEV = [11, 22, 33]
TRAIN_SEEDS_FINAL_EXTRA = [44, 55]
STRATA = ["iid", "gain_ood", "bias_ood", "joint_ood"]
STRATUM_INDEX = {name: index for index, name in enumerate(STRATA)}
SUPPORT_S = 3
INNER_U_VALUES = [0, 1, 3, 5]
QUERY_HORIZON = 10
QUERY_REPEATS = 8


@dataclass(frozen=True)
class P17RetrofitConfig:
    config_id: str = "p17_history256_physical_latent_correction"
    updates: int = 1500
    batch_size: int = 64
    support_s_train: int = SUPPORT_S
    gradient_u_train: int = 3
    query_horizon: int = QUERY_HORIZON
    lr: float = 1.0e-3
    inner_alpha_init: float = 0.5
    lambda_z: float = 1.0e-4
    residual_reg: float = 1.0e-3
    support_prediction_weight: float = 1.0
    train_stream_offset: int = 65_000_000


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


class P17RetrofittedHistorySystem(nn.Module):
    def __init__(self, history: P15HistorySystem, *, inner_alpha_init: float = 0.5):
        super().__init__()
        self.history = history
        self.predictor = PhysicalDisplacementPredictor()
        self.log_inner_alpha = nn.Parameter(torch.log(torch.tensor(float(inner_alpha_init))))
        self.freeze_history()

    @property
    def inner_alpha(self) -> Tensor:
        return self.log_inner_alpha.exp()

    def freeze_history(self) -> None:
        self.history.eval()
        for parameter in self.history.parameters():
            parameter.requires_grad_(False)

    def freeze_all(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.predictor.parameters()
        yield self.log_inner_alpha


def load_history_checkpoint(checkpoint: Path, *, device: torch.device) -> tuple[P15HistorySystem, Dict[str, object]]:
    from tools.fast_adapter.p16_experiment import load_history

    history, train_config, _cfg = load_history(checkpoint, device=device)
    history.eval()
    for parameter in history.parameters():
        parameter.requires_grad_(False)
    return history, train_config


def build_retrofit_from_history_checkpoint(
    history_checkpoint: Path,
    *,
    device: torch.device,
    inner_alpha_init: float = 0.5,
) -> tuple[P17RetrofittedHistorySystem, Dict[str, object]]:
    history, train_config = load_history_checkpoint(history_checkpoint, device=device)
    model = P17RetrofittedHistorySystem(history, inner_alpha_init=inner_alpha_init).to(device)
    return model, train_config


def retrofit_state_dict(model: P17RetrofittedHistorySystem) -> Dict[str, object]:
    return {
        "predictor": model.predictor.state_dict(),
        "log_inner_alpha": model.log_inner_alpha.detach().cpu(),
    }


def load_retrofit(checkpoint: Path, *, device: torch.device, freeze_all: bool) -> tuple[P17RetrofittedHistorySystem, Dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = P17RetrofitConfig(**ckpt["config"]["retrofit_config"])
    history_checkpoint = Path(ckpt["config"]["history_checkpoint"])
    expected_history_sha = ckpt["config"].get("history_checkpoint_sha256")
    actual_history_sha = sha256_file(history_checkpoint)
    if expected_history_sha is not None and actual_history_sha != expected_history_sha:
        raise ValueError(
            f"history checkpoint sha256 mismatch for {history_checkpoint}: "
            f"expected {expected_history_sha}, got {actual_history_sha}"
        )
    model, _history_config = build_retrofit_from_history_checkpoint(history_checkpoint, device=device, inner_alpha_init=cfg.inner_alpha_init)
    model.predictor.load_state_dict(ckpt["retrofit"]["predictor"])
    model.log_inner_alpha.data.copy_(ckpt["retrofit"]["log_inner_alpha"].to(device))
    if freeze_all:
        model.freeze_all()
    else:
        model.freeze_history()
        for parameter in model.predictor.parameters():
            parameter.requires_grad_(True)
        model.log_inner_alpha.requires_grad_(True)
    return model, ckpt["config"]


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_p17_domains(num_domains: int, seed: int, config: Toy2DConfig, *, stratum: str, device: torch.device) -> Toy2DDomain:
    if stratum == "iid":
        return sample_toy2d_domains(num_domains, seed, config, device=device)
    if stratum not in STRATUM_INDEX:
        raise ValueError(f"unknown P17 stratum: {stratum}")
    generator = torch.Generator(device=device).manual_seed(seed)
    iid_gain = torch.empty((num_domains, 2), device=device).uniform_(config.gain_min, config.gain_max, generator=generator)
    iid_bias = torch.empty((num_domains, 2), device=device).uniform_(config.bias_min, config.bias_max, generator=generator)
    low_gain = torch.empty((num_domains, 2), device=device).uniform_(0.6, 0.75, generator=generator)
    high_gain = torch.empty((num_domains, 2), device=device).uniform_(1.25, 1.4, generator=generator)
    gain_side = torch.randint(0, 2, (num_domains, 2), generator=generator, device=device, dtype=torch.bool)
    gain_ood = torch.where(gain_side, high_gain, low_gain)
    bias_mag = torch.empty((num_domains, 2), device=device).uniform_(0.075, 0.09, generator=generator)
    bias_sign = torch.where(
        torch.randint(0, 2, (num_domains, 2), generator=generator, device=device, dtype=torch.bool),
        torch.ones((num_domains, 2), device=device),
        -torch.ones((num_domains, 2), device=device),
    )
    bias_ood = bias_sign * bias_mag
    if stratum == "gain_ood":
        gain, bias = gain_ood, iid_bias
    elif stratum == "bias_ood":
        gain, bias = iid_gain, bias_ood
    else:
        gain, bias = gain_ood, bias_ood
    return Toy2DDomain(gain=gain, bias=bias, domain_id=torch.arange(num_domains, device=device, dtype=torch.long))


def support_physical_fingerprint(records: Sequence[SupportRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        for tensor in [record.state, record.action, record.next_state]:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def support_prediction_terms(
    model: P17RetrofittedHistorySystem,
    records: Sequence[SupportRecord],
    z: Tensor,
    z_init: Tensor,
    cfg: P17RetrofitConfig,
) -> Dict[str, Tensor]:
    if not records:
        zeros = torch.zeros(z.shape[0], device=z.device)
        return {"mse": zeros, "z_reg": zeros, "total": zeros}
    mse_sum = torch.zeros(z.shape[0], device=z.device)
    for record in records:
        pred = model.predictor(record.u.detach(), record.action.detach(), z)
        target = record.delta_x.detach()
        mse_sum = mse_sum + (pred - target).square().mean(dim=-1)
    mse = mse_sum / float(len(records))
    z_reg = (z - z_init.detach()).square().mean(dim=-1)
    return {"mse": mse, "z_reg": z_reg, "total": mse + cfg.lambda_z * z_reg}


def terms_to_float(prefix: str, terms: Dict[str, Tensor]) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value.detach().mean().cpu()) for key, value in terms.items()}


def inner_update_full_batch(
    model: P17RetrofittedHistorySystem,
    records: Sequence[SupportRecord],
    z: Tensor,
    z_init: Tensor,
    cfg: P17RetrofitConfig,
    *,
    create_graph: bool,
) -> tuple[Tensor, Dict[str, float]]:
    pre = support_prediction_terms(model, records, z, z_init, cfg)
    loss = pre["total"].sum()
    grad_z = torch.autograd.grad(loss, z, create_graph=create_graph, retain_graph=create_graph, only_inputs=True)[0]
    z_next = z - model.inner_alpha * grad_z
    if create_graph:
        z_next.retain_grad()
    post = support_prediction_terms(model, records, z_next, z_init, cfg)
    metrics = terms_to_float("pre", pre)
    metrics.update(terms_to_float("post", post))
    metrics["grad_z_norm"] = float(torch.linalg.norm(grad_z.detach(), dim=-1).mean().cpu())
    return z_next, metrics


def adapt_history_latent(
    model: P17RetrofittedHistorySystem,
    records: Sequence[SupportRecord],
    z_init: Tensor,
    *,
    u_passes: int,
    cfg: P17RetrofitConfig,
    create_graph: bool,
) -> tuple[Tensor, List[Dict[str, float]]]:
    if u_passes < 0:
        raise ValueError("U must be non-negative")
    z = z_init.detach().clone().requires_grad_(True)
    metrics: List[Dict[str, float]] = []
    for pass_index in range(u_passes):
        z, step = inner_update_full_batch(model, records, z, z_init.detach(), cfg, create_graph=create_graph)
        step.update({"pass_index": pass_index, "support_records": len(records)})
        metrics.append(step)
    return z, metrics


def history_action_from_z(
    model: P17RetrofittedHistorySystem,
    obs: Toy2DObservation,
    z: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    u = model.history.encode(obs)
    base = model.history.base_action(obs, u)
    delta = model.history.adapter(u, z, base)
    return base + delta, delta, base, u


def query_loss_corrected_history(
    model: P17RetrofittedHistorySystem,
    env: Toy2DVectorEnv,
    domain: Toy2DDomain,
    obs: Toy2DObservation,
    z: Tensor,
    *,
    horizon: int,
    residual_reg: float,
) -> tuple[Tensor, Dict[str, float]]:
    action_losses = []
    residual_losses = []
    first_mse = None
    for _ in range(horizon):
        action, delta, _base, _u = history_action_from_z(model, obs, z)
        executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
        expert = env.expert_action(obs, domain)
        mse = (executed - expert).square().mean(dim=-1)
        if first_mse is None:
            first_mse = float(mse.detach().mean().cpu())
        action_losses.append(mse.mean())
        residual_losses.append(delta.square().mean())
        obs = env.step(obs, domain, executed.detach())
    final_error = env.goal_error(obs).detach()
    success = env.success(obs).detach()
    loss = torch.stack(action_losses).mean() + residual_reg * torch.stack(residual_losses).mean()
    return loss, {
        "first_query_action_mse": float(first_mse),
        "query_action_mse": float(torch.stack([item.detach() for item in action_losses]).mean().cpu()),
        "final_goal_error": float(final_error.mean().cpu()),
        "success_rate": float(success.float().mean().cpu()),
    }


def support_prediction_outer_mse(
    model: P17RetrofittedHistorySystem,
    records: Sequence[SupportRecord],
    z: Tensor,
    z_init: Tensor,
    cfg: P17RetrofitConfig,
) -> Tensor:
    return support_prediction_terms(model, records, z, z_init, cfg)["mse"].mean()


def compute_z_init(model: P17RetrofittedHistorySystem, records: Sequence[SupportRecord], *, batch_size: int, device: torch.device) -> Tensor:
    with torch.no_grad():
        return history_update_context(model.history, records, s=SUPPORT_S, batch_size=batch_size, device=device).detach()


def train_retrofit(args: argparse.Namespace) -> Dict[str, object]:
    cfg = P17RetrofitConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        support_s_train=args.support_s_train,
        gradient_u_train=args.gradient_u_train,
        query_horizon=args.query_horizon,
        train_stream_offset=args.train_stream_offset,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.train_seed)
    toy = Toy2DConfig()
    env = Toy2DVectorEnv(toy, device=device)
    history_checkpoint = Path(args.history_checkpoint)
    model, history_config = build_retrofit_from_history_checkpoint(history_checkpoint, device=device, inner_alpha_init=cfg.inner_alpha_init)
    history_hash_before = tensor_state_hash(model.history)
    opt = torch.optim.Adam(list(model.trainable_parameters()), lr=cfg.lr)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": "p17_retrofit_history_physical",
        "train_seed": args.train_seed,
        "retrofit_config": asdict(cfg),
        "toy_config": toy.to_dict(),
        "history_checkpoint": str(history_checkpoint),
        "history_checkpoint_sha256": sha256_file(history_checkpoint),
        "history_train_config": history_config,
        "param_count": {
            "history": count_parameters(model.history),
            "predictor": count_parameters(model.predictor),
            "retrofit_trainable": sum(p.numel() for p in model.trainable_parameters() if p.requires_grad),
        },
        "compute_note": "P17 adds predictor/log-alpha training on top of the sealed P16 history checkpoint; it is not training-budget-matched against the original history baseline.",
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    records_out = []
    started = time.perf_counter()
    with (outdir / "history.jsonl").open("w") as handle:
        for update_idx in range(cfg.updates):
            step_start = time.perf_counter()
            stream = cfg.train_stream_offset + args.train_seed * 100000 + update_idx
            domain = sample_toy2d_domains(cfg.batch_size, stream, toy, device=device)
            support_records = collect_common_base_records(model.history, env, domain, support_seed=stream + 10_000_000, max_s=cfg.support_s_train)
            z_init = compute_z_init(model, support_records, batch_size=cfg.batch_size, device=device)
            z, support_metrics = adapt_history_latent(model, support_records, z_init, u_passes=cfg.gradient_u_train, cfg=cfg, create_graph=True)
            query_obs = env.reset(cfg.batch_size, stream + 20_000_000, domain=domain)
            action_loss, query_metrics = query_loss_corrected_history(model, env, domain, query_obs, z, horizon=cfg.query_horizon, residual_reg=cfg.residual_reg)
            support_outer = support_prediction_outer_mse(model, support_records, z, z_init, cfg)
            loss = action_loss + cfg.support_prediction_weight * support_outer
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = {
                "predictor": grad_norm(model.predictor.parameters()),
                "log_inner_alpha": grad_norm([model.log_inner_alpha]),
                "history": grad_norm(model.history.parameters()),
            }
            opt.step()
            finite = bool(torch.isfinite(loss.detach()).item() and torch.isfinite(model.inner_alpha.detach()).item())
            rec = {
                "update": update_idx + 1,
                "loss": float(loss.detach().cpu()),
                "action_loss": float(action_loss.detach().cpu()),
                "support_prediction_mse": float(support_outer.detach().cpu()),
                "finite": finite,
                "inner_alpha": float(model.inner_alpha.detach().cpu()),
                "z_delta_norm": float(torch.linalg.norm(z.detach() - z_init.detach(), dim=-1).mean().cpu()),
                "support_physical_fingerprint": support_physical_fingerprint(support_records),
                "sec_update": time.perf_counter() - step_start,
                "support_metrics": support_metrics,
                **query_metrics,
                "grad_norms": grads,
            }
            records_out.append(rec)
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            if not finite:
                break
            if args.log_every and (update_idx == 0 or (update_idx + 1) % args.log_every == 0):
                print(json.dumps(rec, sort_keys=True))
    elapsed = time.perf_counter() - started
    ckpt_path = outdir / "checkpoint.pt"
    torch.save({"retrofit": retrofit_state_dict(model), "config": config, "history_tail": records_out[-10:]}, ckpt_path)
    history_hash_after = tensor_state_hash(model.history)
    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "elapsed_sec": elapsed,
        "mean_sec_update": elapsed / max(1, len(records_out)),
        "completed_updates": len(records_out),
        "final_parameter_allfinite": all(torch.isfinite(p.detach()).all().item() for p in model.parameters()),
        "history_freeze_audit": {"before": history_hash_before, "after": history_hash_after, "unchanged": history_hash_before == history_hash_after},
        "extra_train_cost": {"updates": len(records_out), "batch_size": cfg.batch_size, "elapsed_sec": elapsed},
        "last": records_out[-1] if records_out else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def calibration_records(model: P17RetrofittedHistorySystem, env: Toy2DVectorEnv, domain: Toy2DDomain, *, seed: int, starts: int) -> List[SupportRecord]:
    records: List[SupportRecord] = []
    for start_index in range(starts):
        records.extend(collect_common_base_records(model.history, env, domain, support_seed=seed + start_index, max_s=1))
    return records


def heldout_prediction_mse(model: P17RetrofittedHistorySystem, records: Sequence[SupportRecord], z: Tensor) -> Tensor:
    losses = []
    with torch.no_grad():
        for record in records:
            pred = model.predictor(record.u, record.action, z)
            losses.append((pred - record.delta_x).square().mean(dim=-1))
    return torch.stack(losses).mean(dim=0) if losses else torch.full((z.shape[0],), float("nan"), device=z.device)


def make_swapped_correction(z_init: Tensor, z_adapted: Tensor, perm: Tensor) -> Tensor:
    return z_init + (z_adapted - z_init)[perm]


def rollout_history_z_arrays(
    model: P17RetrofittedHistorySystem,
    env: Toy2DVectorEnv,
    domain: Toy2DDomain,
    obs: Toy2DObservation,
    z: Tensor,
    z_init: Tensor,
    *,
    arm: str,
    horizon: int,
) -> Dict[str, Tensor]:
    first_mse = None
    first_change = None
    mse_steps = []
    delta_norm = torch.zeros(obs.state.shape[0], device=obs.state.device)
    action_abs_max = torch.zeros(obs.state.shape[0], device=obs.state.device)
    with torch.no_grad():
        init_action, _init_delta, _init_base, _init_u = history_action_from_z(model, obs, z_init)
        init_executed = torch.clamp(init_action, -env.config.action_limit, env.config.action_limit)
        for _step in range(horizon):
            if arm == "nominal_base":
                action = env.nominal_action(obs)
                delta_norm = torch.zeros(action.shape[0], device=action.device)
            else:
                action, delta, _base, _u = history_action_from_z(model, obs, z)
                delta_norm = torch.linalg.norm(delta, dim=-1)
            executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
            action_abs_max = torch.maximum(action_abs_max, executed.detach().abs().amax(dim=-1))
            expert = env.expert_action(obs, domain)
            mse = (executed - expert).square().mean(dim=-1)
            if first_mse is None:
                first_mse = mse
                first_change = torch.linalg.norm(executed - init_executed, dim=-1)
            mse_steps.append(mse)
            obs = env.step(obs, domain, executed)
    return {
        "first_mse": first_mse,
        "first_action_change_vs_history_init": first_change,
        "query_mse": torch.stack(mse_steps).mean(dim=0),
        "final_error": env.goal_error(obs),
        "success": env.success(obs),
        "delta_norm": delta_norm,
        "action_abs_max": action_abs_max,
    }


def rollout_ls_arrays_p17(
    env: Toy2DVectorEnv,
    domain: Toy2DDomain,
    obs: Toy2DObservation,
    fit: Dict[str, Tensor],
    init_executed: Tensor,
    *,
    horizon: int,
) -> Dict[str, Tensor]:
    first_mse = None
    first_change = None
    mse_steps = []
    action_abs_max = torch.zeros(obs.state.shape[0], device=obs.state.device)
    saturation_steps = []
    with torch.no_grad():
        identifiable = (fit["rank"] >= 2).all(dim=-1).to(obs.state.device)
        for _step in range(horizon):
            base = env.nominal_action(obs)
            gain = torch.clamp(fit["gain_hat"].to(base.device), 0.1, 10.0)
            bias = fit["bias_hat"].to(base.device)
            corrected = (base - bias) / gain
            residual = torch.clamp(corrected - base, -0.15, 0.15)
            action = torch.clamp(base + residual, -env.config.action_limit, env.config.action_limit)
            action = torch.where(identifiable.unsqueeze(-1), action, base)
            action_abs_max = torch.maximum(action_abs_max, action.detach().abs().amax(dim=-1))
            expert = env.expert_action(obs, domain)
            mse = (action - expert).square().mean(dim=-1)
            if first_mse is None:
                first_mse = mse
                first_change = torch.linalg.norm(action - init_executed, dim=-1)
            mse_steps.append(mse)
            saturation_steps.append((residual.abs() >= 0.15 - 1.0e-6).any(dim=-1).float() * identifiable.float())
            obs = env.step(obs, domain, action)
    return {
        "first_mse": first_mse,
        "first_action_change_vs_history_init": first_change,
        "query_mse": torch.stack(mse_steps).mean(dim=0),
        "final_error": env.goal_error(obs),
        "success": env.success(obs),
        "action_abs_max": action_abs_max,
        "saturation_fraction": torch.stack(saturation_steps).mean(dim=0),
        "identifiable": identifiable,
    }


def pack_query_rows(
    arrays: Dict[str, Tensor],
    *,
    meta: Dict[str, object],
    arm: str,
    u_passes: Optional[int],
    domain_ids: Tensor,
    query_repeats: int,
    z_source_domain: Optional[Tensor],
    z_delta_norm: Optional[Tensor],
    support_loss_pre: Optional[Tensor],
    support_loss_post: Optional[Tensor],
    heldout_mse: Optional[Tensor],
) -> List[Dict[str, object]]:
    rows = []
    num_domains = domain_ids.shape[0] // query_repeats
    cpu = {key: value.detach().cpu() for key, value in arrays.items()}
    ids = domain_ids.detach().cpu()
    source = None if z_source_domain is None else z_source_domain.detach().cpu()
    zdelta = None if z_delta_norm is None else z_delta_norm.detach().cpu()
    pre = None if support_loss_pre is None else support_loss_pre.detach().cpu()
    post = None if support_loss_post is None else support_loss_post.detach().cpu()
    heldout = None if heldout_mse is None else heldout_mse.detach().cpu()
    for domain_row in range(num_domains):
        start = domain_row * query_repeats
        end = start + query_repeats
        source_idx = None if source is None else int(source[start])
        rows.append({
            **meta,
            "method": "p17_retrofit_history",
            "arm": arm,
            "S": SUPPORT_S,
            "U": u_passes,
            "support_mode": "common_base",
            "domain_index": int(ids[start]),
            "query_indices": list(range(query_repeats)),
            "z_source_domain_index": source_idx,
            "final_goal_errors": [float(x) for x in cpu["final_error"][start:end]],
            "successes": [bool(x) for x in cpu["success"][start:end]],
            "first_query_action_mse": [float(x) for x in cpu["first_mse"][start:end]],
            "first_action_change_vs_history_init": [float(x) for x in cpu["first_action_change_vs_history_init"][start:end]],
            "query_action_mse": [float(x) for x in cpu["query_mse"][start:end]],
            "delta_norm_last": [float(x) for x in cpu["delta_norm"][start:end]] if "delta_norm" in cpu else None,
            "action_abs_max": [float(x) for x in cpu["action_abs_max"][start:end]],
            "z_delta_norm": None if zdelta is None or source_idx is None else float(zdelta[source_idx]),
            "support_prediction_loss_pre": None if pre is None else float(pre[domain_row]),
            "support_prediction_loss_post": None if post is None else float(post[domain_row]),
            "heldout_physical_prediction_mse": None if heldout is None else float(heldout[domain_row]),
        })
    return rows


def pack_ls_rows(
    arrays: Dict[str, Tensor],
    *,
    meta: Dict[str, object],
    domain_ids: Tensor,
    query_repeats: int,
    fit: Dict[str, Tensor],
    heldout_mse: Tensor,
) -> List[Dict[str, object]]:
    rows = []
    num_domains = domain_ids.shape[0] // query_repeats
    cpu = {key: value.detach().cpu() for key, value in arrays.items()}
    ids = domain_ids.detach().cpu()
    rank = fit["rank"].detach().cpu()
    condition = fit["condition"].detach().cpu()
    sv = fit["singular_values"].detach().cpu()
    heldout = heldout_mse.detach().cpu()
    for domain_row in range(num_domains):
        start = domain_row * query_repeats
        end = start + query_repeats
        rows.append({
            **meta,
            "method": "bounded_least_squares",
            "arm": "bounded_ls_s3",
            "S": SUPPORT_S,
            "U": None,
            "support_mode": "common_base",
            "domain_index": int(ids[start]),
            "query_indices": list(range(query_repeats)),
            "identifiable": bool(cpu["identifiable"][start]),
            "rank": [int(x) for x in rank[domain_row]],
            "condition": [float(x) for x in condition[domain_row]],
            "singular_values": sv[domain_row].tolist(),
            "final_goal_errors": [float(x) for x in cpu["final_error"][start:end]],
            "successes": [bool(x) for x in cpu["success"][start:end]],
            "first_query_action_mse": [float(x) for x in cpu["first_mse"][start:end]],
            "first_action_change_vs_history_init": [float(x) for x in cpu["first_action_change_vs_history_init"][start:end]],
            "query_action_mse": [float(x) for x in cpu["query_mse"][start:end]],
            "action_abs_max": [float(x) for x in cpu["action_abs_max"][start:end]],
            "saturation_fraction": [float(x) for x in cpu["saturation_fraction"][start:end]],
            "heldout_physical_prediction_mse": float(heldout[domain_row]) if torch.isfinite(heldout[domain_row]) else None,
        })
    return rows


def support_loss_vector(model: P17RetrofittedHistorySystem, records: Sequence[SupportRecord], z: Tensor, z_init: Tensor, cfg: P17RetrofitConfig) -> Tensor:
    return support_prediction_terms(model, records, z, z_init, cfg)["total"].detach()


def evaluate_retrofit(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint)
    model, train_config = load_retrofit(checkpoint, device=device, freeze_all=True)
    before = tensor_state_hash(model)
    toy = Toy2DConfig()
    env = Toy2DVectorEnv(toy, device=device)
    cfg = P17RetrofitConfig(**train_config["retrofit_config"])
    rows: List[Dict[str, object]] = []
    support_rows: List[Dict[str, object]] = []
    heldout_rows: List[Dict[str, object]] = []
    timing_rows: List[Dict[str, object]] = []
    for stratum in args.strata:
        stratum_index = STRATUM_INDEX[stratum]
        domain_seed = args.domain_seed + stratum_index * 1000
        support_seed = args.support_seed + stratum_index * 1000
        calibration_seed = args.calibration_seed + stratum_index * 1000
        query_seed = args.query_seed + stratum_index * 1000
        domain = sample_p17_domains(args.num_domains, domain_seed, toy, stratum=stratum, device=device)
        query_domain = repeat_domain(domain, args.query_repeats)
        support_records = collect_common_base_records(model.history, env, domain, support_seed=support_seed, max_s=SUPPORT_S)
        support_fingerprint = support_physical_fingerprint(support_records)
        calibration = calibration_records(model, env, domain, seed=calibration_seed, starts=args.calibration_starts)
        _sync_for_timing(device)
        timing_start = time.perf_counter()
        z_init = compute_z_init(model, support_records, batch_size=args.num_domains, device=device)
        _sync_for_timing(device)
        timing_rows.append({
            "split": args.split,
            "train_seed": args.train_seed,
            "operation": "history_init",
            "stratum": stratum,
            "stratum_index": stratum_index,
            "S": SUPPORT_S,
            "U": 0,
            "batched_domains": args.num_domains,
            "elapsed_sec": time.perf_counter() - timing_start,
            "overhead_note": "Includes batched frozen-history GRU context construction only; excludes support collection, rollouts, diagnostics, and file I/O.",
        })
        z_by_u: Dict[int, Tensor] = {0: z_init.detach()}
        metrics_by_u: Dict[int, List[Dict[str, float]]] = {0: []}
        for u_passes in sorted(set(args.u_values) - {0}):
            _sync_for_timing(device)
            timing_start = time.perf_counter()
            with torch.enable_grad():
                z, metrics = adapt_history_latent(model, support_records, z_init, u_passes=u_passes, cfg=cfg, create_graph=False)
            _sync_for_timing(device)
            timing_rows.append({
                "split": args.split,
                "train_seed": args.train_seed,
                "operation": "latent_correction",
                "stratum": stratum,
                "stratum_index": stratum_index,
                "S": SUPPORT_S,
                "U": u_passes,
                "batched_domains": args.num_domains,
                "elapsed_sec": time.perf_counter() - timing_start,
                "overhead_note": "Includes the requested batched full support passes for this U; excludes rollouts, heldout/LS diagnostics, support collection, and file I/O.",
            })
            z_by_u[u_passes] = z.detach()
            metrics_by_u[u_passes] = metrics
        perm = roll1_derangement(args.num_domains, device=device)
        z3_swapped = make_swapped_correction(z_init, z_by_u[3], perm) if 3 in z_by_u else None
        meta = {
            "split": args.split,
            "train_seed": args.train_seed,
            "config_id": train_config["retrofit_config"]["config_id"],
            "stratum": stratum,
            "stratum_index": stratum_index,
            "domain_seed": domain_seed,
            "support_seed": support_seed,
            "calibration_seed": calibration_seed,
            "query_seed": query_seed,
            "query_repeats": args.query_repeats,
            "support_physical_fingerprint": support_fingerprint,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "history_checkpoint": train_config["history_checkpoint"],
            "history_checkpoint_sha256": train_config["history_checkpoint_sha256"],
        }
        for u_passes, metrics in metrics_by_u.items():
            for item in metrics:
                support_rows.append({**meta, "S": SUPPORT_S, "U": u_passes, **item})
        z_init_rep = z_init.repeat_interleave(args.query_repeats, dim=0)
        identity = torch.arange(args.num_domains, device=device).repeat_interleave(args.query_repeats)
        arm_specs: List[tuple[str, Optional[int], Tensor, Tensor, Optional[Tensor]]] = [
            ("nominal_base", None, z_init, identity, torch.zeros(args.num_domains, device=device)),
            ("history_init", 0, z_init, identity, torch.zeros(args.num_domains, device=device)),
        ]
        for u_passes in args.u_values:
            if u_passes == 0:
                continue
            z_u = z_by_u[u_passes]
            arm_specs.append((f"corrected_u{u_passes}", u_passes, z_u, identity, torch.linalg.norm(z_u - z_init, dim=-1)))
        if z3_swapped is not None:
            arm_specs.append(("swapped_correction_u3", 3, z3_swapped, perm.repeat_interleave(args.query_repeats), torch.linalg.norm(z_by_u[3] - z_init, dim=-1)))
        for arm, u_passes, z_arm, source, z_delta in arm_specs:
            z_arm_rep = z_arm.repeat_interleave(args.query_repeats, dim=0)
            qobs = env.reset(args.num_domains * args.query_repeats, query_seed, domain=query_domain)
            arrays = rollout_history_z_arrays(model, env, query_domain, qobs, z_arm_rep, z_init_rep, arm=arm, horizon=args.query_horizon)
            pre_loss = support_loss_vector(model, support_records, z_init, z_init, cfg) if u_passes is not None else None
            post_loss = support_loss_vector(model, support_records, z_arm, z_init, cfg) if u_passes is not None else None
            heldout = heldout_prediction_mse(model, calibration, z_arm) if u_passes is not None else None
            rows.extend(pack_query_rows(arrays, meta=meta, arm=arm, u_passes=u_passes, domain_ids=query_domain.domain_id, query_repeats=args.query_repeats, z_source_domain=source, z_delta_norm=z_delta, support_loss_pre=pre_loss, support_loss_post=post_loss, heldout_mse=heldout))
            if heldout is not None:
                for domain_index, value in enumerate(heldout.detach().cpu()):
                    heldout_rows.append({**meta, "method": "p17_retrofit_history", "arm": arm, "S": SUPPORT_S, "U": u_passes, "domain_index": domain_index, "heldout_physical_prediction_mse": float(value)})
        fit = ls_fit_delta(support_records[:SUPPORT_S])
        ls_cal_losses = []
        for record in calibration:
            pred = fit["gain_hat"].to(record.action.device) * record.action + fit["bias_hat"].to(record.action.device)
            mse = (pred - record.delta_x).square().mean(dim=-1)
            identifiable = (fit["rank"] >= 2).all(dim=-1).to(mse.device)
            ls_cal_losses.append(torch.where(identifiable, mse, torch.full_like(mse, float("nan"))))
        ls_heldout = torch.stack(ls_cal_losses).nanmean(dim=0) if ls_cal_losses else torch.full((args.num_domains,), float("nan"), device=device)
        qobs = env.reset(args.num_domains * args.query_repeats, query_seed, domain=query_domain)
        with torch.no_grad():
            init_action, _delta, _base, _u = history_action_from_z(model, qobs, z_init_rep)
            init_executed = torch.clamp(init_action, -env.config.action_limit, env.config.action_limit)
        fit_rep = {key: value.repeat_interleave(args.query_repeats, dim=0) for key, value in fit.items()}
        ls_arrays = rollout_ls_arrays_p17(env, query_domain, qobs, fit_rep, init_executed, horizon=args.query_horizon)
        rows.extend(pack_ls_rows(ls_arrays, meta=meta, domain_ids=query_domain.domain_id, query_repeats=args.query_repeats, fit=fit, heldout_mse=ls_heldout))
    after = tensor_state_hash(model)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "eval_raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    support_path = outdir / "support_transition_metrics.jsonl"
    support_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in support_rows))
    heldout_path = outdir / "heldout_physical_prediction_mse.jsonl"
    heldout_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in heldout_rows))
    timing_path = outdir / "eval_timing.jsonl"
    timing_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in timing_rows))
    summary = {
        "raw_path": str(raw_path),
        "support_transition_metrics_path": str(support_path),
        "heldout_physical_prediction_mse_path": str(heldout_path),
        "eval_timing_path": str(timing_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_config": train_config,
        "freeze_audit": {"before": before, "after": after, "unchanged": before == after},
        "schema_note": "eval_raw rows are per domain with full query arrays; support fingerprints hash state/action/next_state only and exclude encoded u.",
    }
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def protocol() -> Dict[str, object]:
    return {
        "status": "prepared_not_formally_run",
        "p16_history_checkpoints": {str(seed): str(path) for seed, path in P16_HISTORY_CHECKPOINTS.items()},
        "dev_train_seeds": TRAIN_SEEDS_DEV,
        "conditional_final_extra_seeds": TRAIN_SEEDS_FINAL_EXTRA,
        "train": asdict(P17RetrofitConfig()),
        "eval": {
            "strata": STRATA,
            "support_S": SUPPORT_S,
            "arms": ["nominal_base", "history_init", "corrected_u1", "corrected_u3", "corrected_u5", "swapped_correction_u3", "bounded_ls_s3"],
            "query_horizon": QUERY_HORIZON,
            "query_repeats": QUERY_REPEATS,
            "dev": {"num_domains": 128, "domain_seed": 710001, "support_seed": 720001, "calibration_seed": 730001, "query_seed": 740001},
            "final": {"num_domains": 256, "domain_seed": 810001, "support_seed": 820001, "calibration_seed": 830001, "query_seed": 840001},
            "stratum_seed_offset": "base + stratum_index * 1000, iid=0 gain_ood=1 bias_ood=2 joint_ood=3",
        },
        "gate": {
            "dev": "all 3 seeds joint_ood corrected_u3-vs-history_init SR>0 and error<0, mean SR>=2pp; IID mean SR>=-1pp; bounded LS joint SR>=95%; otherwise close dev negative/mixed with no tuning/final",
            "final": "paired bootstrap 5000 seed 20260922 over model seed and domain axes; joint SR CI lower>0, error CI upper<0, >=4/5 jointly positive, mean SR>=2pp, IID mean>=-1pp, LS joint>=95%",
        },
    }


def write_protocol(args: argparse.Namespace) -> Dict[str, object]:
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = protocol()
    path = outdir / "P17_PREPARED_PROTOCOL.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {"protocol_path": str(path), **payload}


def counts(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device("cpu")
    history = P15HistorySystem(__import__("tools.fast_adapter.p15_experiment", fromlist=["HistoryConfig"]).HistoryConfig(config_id="history", width=256)).to(device)
    model = P17RetrofittedHistorySystem(history)
    result = {
        "history": count_parameters(model.history),
        "predictor": count_parameters(model.predictor),
        "retrofit_trainable": sum(p.numel() for p in model.trainable_parameters() if p.requires_grad),
        "protocol": "Counts exclude sealed P16 history training and include only new predictor/log-alpha trainable parameters for P17 extra cost.",
    }
    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "P17_PARAM_COUNTS.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history-checkpoint", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-s-train", type=int, default=SUPPORT_S)
    parser.add_argument("--gradient-u-train", type=int, default=3)
    parser.add_argument("--query-horizon", type=int, default=QUERY_HORIZON)
    parser.add_argument("--train-stream-offset", type=int, default=65_000_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=500)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-id", default=P17RetrofitConfig().config_id)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--split", choices=["dev", "final", "smoke"], default="dev")
    parser.add_argument("--domain-seed", type=int, required=True)
    parser.add_argument("--support-seed", type=int, required=True)
    parser.add_argument("--calibration-seed", type=int, required=True)
    parser.add_argument("--query-seed", type=int, required=True)
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--query-repeats", type=int, default=QUERY_REPEATS)
    parser.add_argument("--query-horizon", type=int, default=QUERY_HORIZON)
    parser.add_argument("--calibration-starts", type=int, default=4)
    parser.add_argument("--u-values", type=int, nargs="+", default=INNER_U_VALUES)
    parser.add_argument("--strata", nargs="+", choices=STRATA, default=STRATA)
    parser.add_argument("--device", default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P17 bounded history-latent physical correction experiment.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train-retrofit")
    add_train_args(train)
    train.set_defaults(func=train_retrofit)
    evaluate = sub.add_parser("eval-retrofit")
    add_eval_args(evaluate)
    evaluate.set_defaults(func=evaluate_retrofit)
    proto = sub.add_parser("write-protocol")
    proto.add_argument("--output-dir", default=str(P17_ROOT))
    proto.set_defaults(func=write_protocol)
    count_parser = sub.add_parser("counts")
    count_parser.add_argument("--output-dir", default=None)
    count_parser.set_defaults(func=counts)
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    result = ns.func(ns)
    print(json.dumps(result, indent=2, sort_keys=True))
