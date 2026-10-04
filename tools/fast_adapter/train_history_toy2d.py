from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch

from prove.fast_adapter import HistoryConditionedAdapter, InteractionEncoder, ScriptedNominalPolicy2D, Toy2DConfig, Toy2DVectorEnv, sample_toy2d_domains


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().square().sum().cpu())
    return math.sqrt(total)


class HistoryBaselineSystem(torch.nn.Module):
    def __init__(self, config: Toy2DConfig):
        super().__init__()
        self.encoder = InteractionEncoder(obj_dim=2, tar_dim=2, proprio_dim=2, action_dim=2)
        self.base_policy = ScriptedNominalPolicy2D(action_limit=config.action_limit)
        self.adapter = HistoryConditionedAdapter(max_residual=torch.tensor([0.15, 0.15]))
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()

    def encode(self, obs):
        runtime = obs.adapter_observation()
        return self.encoder(runtime.obj_feat, runtime.tar_feat, runtime.proprio, runtime.prev_action)

    def base_action(self, obs, u):
        with torch.no_grad():
            return self.base_policy(obs.adapter_observation(), u)


def support_context(system: HistoryBaselineSystem, env: Toy2DVectorEnv, domain, obs, *, k: int):
    context = system.adapter.initial_context(domain.gain.shape[0], device=domain.gain.device)
    for _ in range(k):
        u = system.encode(obs)
        base = system.base_action(obs, u)
        delta = system.adapter(u, context, base)
        action = base + delta
        next_obs = env.step(obs, domain, torch.clamp(action, -env.config.action_limit, env.config.action_limit).detach())
        u_next = system.encode(next_obs)
        context = system.adapter.update_context(context, u, next_obs.prev_action, u_next)
        obs = next_obs
    return context, obs


def query_loss(system: HistoryBaselineSystem, env: Toy2DVectorEnv, domain, obs, context, *, horizon: int, residual_reg: float):
    action_losses = []
    residual_losses = []
    final_error = None
    success = None
    first_query_action_mse = None
    for step in range(horizon):
        u = system.encode(obs)
        base = system.base_action(obs, u)
        delta = system.adapter(u, context, base)
        action = base + delta
        executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
        expert = env.expert_action(obs, domain)
        mse = (executed - expert).square().mean(dim=-1)
        if step == 0:
            first_query_action_mse = float(mse.mean().detach().cpu())
        action_losses.append(mse.mean())
        residual_losses.append(delta.square().mean())
        obs = env.step(obs, domain, executed.detach())
    final_error_tensor = env.goal_error(obs).detach()
    success_tensor = env.success(obs).detach()
    loss = torch.stack(action_losses).mean() + residual_reg * torch.stack(residual_losses).mean()
    return loss, {
        'query_action_mse': float(torch.stack([x.detach() for x in action_losses]).mean().cpu()),
        'first_query_action_mse': float(first_query_action_mse),
        'final_goal_error': float(final_error_tensor.mean().cpu()),
        'success_rate': float(success_tensor.float().mean().cpu()),
    }


def train(args: argparse.Namespace):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.train_seed)
    cfg = Toy2DConfig()
    env = Toy2DVectorEnv(cfg, device=device)
    system = HistoryBaselineSystem(cfg).to(device)
    opt = torch.optim.Adam([p for p in system.parameters() if p.requires_grad], lr=args.lr)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = {
        'mode': 'history_baseline',
        'train_seed': args.train_seed,
        'updates': args.updates,
        'batch_size': args.batch_size,
        'support_k': args.support_k,
        'query_horizon': args.query_horizon,
        'lr': args.lr,
        'residual_reg': args.residual_reg,
        'toy_config': cfg.to_dict(),
        'runtime_notes': {
            'support': 'on-policy support updates GRU context only; encoder/adapter receive training gradients through support context construction',
            'query': 'context frozen during query; no query context updates',
            'hidden_domain_inputs': 'gain/bias are used only by toy env and expert, never runtime observation',
        },
    }
    (outdir / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True))
    records = []
    started = time.perf_counter()
    with (outdir / 'history.jsonl').open('w') as handle:
        for update_idx in range(args.updates):
            step_start = time.perf_counter()
            domain = sample_toy2d_domains(args.batch_size, args.train_seed * 100000 + update_idx, cfg, device=device)
            support_obs = env.reset(args.batch_size, args.train_seed * 200000 + update_idx, domain=domain)
            context, _ = support_context(system, env, domain, support_obs, k=args.support_k)
            query_obs = env.reset(args.batch_size, args.train_seed * 300000 + update_idx, domain=domain)
            loss, metrics = query_loss(system, env, domain, query_obs, context, horizon=args.query_horizon, residual_reg=args.residual_reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = {
                'encoder': grad_norm(system.encoder.parameters()),
                'adapter': grad_norm(system.adapter.parameters()),
            }
            opt.step()
            rec = {
                'update': update_idx + 1,
                'loss': float(loss.detach().cpu()),
                'sec_update': time.perf_counter() - step_start,
                **metrics,
                'grad_norms': grads,
            }
            records.append(rec)
            handle.write(json.dumps(rec, sort_keys=True) + '\n')
            if args.log_every and ((update_idx + 1) % args.log_every == 0 or update_idx == 0):
                print(json.dumps(rec, sort_keys=True))
    elapsed = time.perf_counter() - started
    ckpt_path = outdir / 'checkpoint.pt'
    torch.save({'system': system.state_dict(), 'config': config, 'history_tail': records[-10:]}, ckpt_path)
    summary = {
        'updates': args.updates,
        'elapsed_sec': elapsed,
        'mean_sec_update': elapsed / max(1, args.updates),
        'last': records[-1] if records else None,
        'checkpoint': str(ckpt_path),
        'history': str(outdir / 'history.jsonl'),
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--updates', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--support-k', type=int, default=3)
    parser.add_argument('--query-horizon', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--residual-reg', type=float, default=1.0e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--log-every', type=int, default=250)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
