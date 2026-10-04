from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch

from prove.fast_adapter import Toy2DConfig, Toy2DDomain, Toy2DVectorEnv, sample_toy2d_domains
from tools.fast_adapter.train_history_toy2d import HistoryBaselineSystem, support_context


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_hash(system) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(system.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def repeat_domain(domain: Toy2DDomain, repeats: int) -> Toy2DDomain:
    return Toy2DDomain(
        gain=domain.gain.repeat_interleave(repeats, dim=0),
        bias=domain.bias.repeat_interleave(repeats, dim=0),
        domain_id=domain.domain_id.repeat_interleave(repeats, dim=0),
    )


def rollout(system, env, domain, obs, context, *, horizon: int, split: str, train_seed: int, domain_seed: int, support_seed: int, query_seed: int, query_repeats: int, support_k: int):
    first = None
    mses = []
    delta_norm = torch.zeros(obs.state.shape[0], device=obs.state.device)
    with torch.no_grad():
        for _ in range(horizon):
            u = system.encode(obs)
            base = system.base_action(obs, u)
            delta = system.adapter(u, context, base)
            delta_norm = torch.linalg.norm(delta, dim=-1)
            action = base + delta
            expert = env.expert_action(obs, domain)
            executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
            mse = (executed - expert).square().mean(dim=-1)
            if first is None:
                first = mse
            mses.append(mse)
            obs = env.step(obs, domain, executed)
        final_error = env.goal_error(obs)
        success = env.success(obs)
        query_mse = torch.stack(mses).mean(dim=0)
    rows = []
    for idx in range(final_error.shape[0]):
        rows.append({
            'split': split,
            'train_seed': train_seed,
            'domain_seed': domain_seed,
            'support_seed': support_seed,
            'query_seed': query_seed,
            'domain_index': int(domain.domain_id[idx].detach().cpu()),
            'query_index': idx % query_repeats,
            'arm': 'history',
            'K': support_k,
            'z_source_domain_index': None,
            'final_goal_error': float(final_error[idx].cpu()),
            'success': bool(success[idx].cpu()),
            'first_query_action_mse': float(first[idx].cpu()),
            'query_action_mse': float(query_mse[idx].cpu()),
            'delta_norm_last': float(delta_norm[idx].cpu()),
            'latent_delta_norm': None,
            'support_loss': None,
        })
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, float]:
    return {
        'n': len(rows),
        'mean_final_goal_error': sum(float(r['final_goal_error']) for r in rows) / len(rows),
        'success_rate': sum(1.0 if r['success'] else 0.0 for r in rows) / len(rows),
        'mean_first_query_action_mse': sum(float(r['first_query_action_mse']) for r in rows) / len(rows),
        'mean_query_action_mse': sum(float(r['query_action_mse']) for r in rows) / len(rows),
    }


def evaluate(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg = Toy2DConfig()
    env = Toy2DVectorEnv(cfg, device=device)
    system = HistoryBaselineSystem(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    system.load_state_dict(ckpt['system'])
    system.eval()
    for p in system.parameters():
        p.requires_grad_(False)
    before = state_hash(system)
    domain = sample_toy2d_domains(args.num_domains, args.domain_seed, cfg, device=device)
    support_seed = args.support_seed
    support_obs = env.reset(args.num_domains, support_seed, domain=domain)
    with torch.no_grad():
        context, _ = support_context(system, env, domain, support_obs, k=args.support_k)
        query_domain = repeat_domain(domain, args.query_repeats)
        query_context = context.repeat_interleave(args.query_repeats, dim=0)
        query_obs = env.reset(args.num_domains * args.query_repeats, args.query_seed, domain=query_domain)
        rows = rollout(system, env, query_domain, query_obs, query_context, horizon=args.query_horizon, split=args.split, train_seed=args.train_seed, domain_seed=args.domain_seed, support_seed=support_seed, query_seed=args.query_seed, query_repeats=args.query_repeats, support_k=args.support_k)
    after = state_hash(system)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / 'eval_raw.jsonl'
    with raw.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    summary = {
        'raw_path': str(raw),
        'summary': {'history:K%s' % args.support_k: summarize(rows)},
        'config': vars(args) | {'checkpoint_sha256': sha(args.checkpoint)},
        'freeze_audit': {'parameter_hash_before_eval': before, 'parameter_hash_after_eval': after, 'unchanged': before == after},
    }
    (outdir / 'eval_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--train-seed', type=int, required=True)
    p.add_argument('--split', default='final')
    p.add_argument('--domain-seed', type=int, default=91000)
    p.add_argument('--support-seed', type=int, default=92000)
    p.add_argument('--query-seed', type=int, default=101000)
    p.add_argument('--num-domains', type=int, default=256)
    p.add_argument('--query-repeats', type=int, default=8)
    p.add_argument('--support-k', type=int, default=3)
    p.add_argument('--query-horizon', type=int, default=10)
    p.add_argument('--device', default=None)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
