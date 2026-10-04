from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch

from prove.fast_adapter import AdaptationLossConfig, Toy2DConfig, Toy2DDomain, Toy2DVectorEnv
from prove.fast_adapter import sample_toy2d_domains
from tools.fast_adapter.train_toy2d import build_system, support_adapt


def repeat_domain(domain: Toy2DDomain, repeats: int) -> Toy2DDomain:
    return Toy2DDomain(
        gain=domain.gain.repeat_interleave(repeats, dim=0),
        bias=domain.bias.repeat_interleave(repeats, dim=0),
        domain_id=domain.domain_id.repeat_interleave(repeats, dim=0),
    )


def tensor_state_hash(system) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(system.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_derangement(n: int, seed: int, device: torch.device) -> torch.Tensor:
    del seed
    if n < 2:
        raise ValueError('shuffle_z diagnostic needs at least two domains')
    return torch.roll(torch.arange(n, device=device), shifts=1)


def rollout_query(system, env, domain, obs, z, *, horizon: int, arm: str, k: int, train_seed: int, split: str, domain_seed: int, support_seed: int, query_seed: int, query_repeats: int, z_source_domain: torch.Tensor, latent_delta: torch.Tensor):
    first_mse = None
    mse_steps = []
    delta_norm = torch.zeros(obs.state.shape[0], device=obs.state.device)
    with torch.no_grad():
        for step in range(horizon):
            if arm == 'base':
                action = env.nominal_action(obs)
                delta_norm = torch.zeros(action.shape[0], device=action.device)
            else:
                action, delta, _base, _u = system.act(obs.adapter_observation(), z)
                delta_norm = torch.linalg.norm(delta, dim=-1)
            expert = env.expert_action(obs, domain)
            executed = torch.clamp(action, -env.config.action_limit, env.config.action_limit)
            mse = (executed - expert).square().mean(dim=-1)
            if step == 0:
                first_mse = mse
            mse_steps.append(mse)
            obs = env.step(obs, domain, executed)
        final_error = env.goal_error(obs)
        success = env.success(obs)
        query_mse = torch.stack(mse_steps, dim=0).mean(dim=0)
    rows = []
    for row_idx in range(final_error.shape[0]):
        domain_index = int(domain.domain_id[row_idx].detach().cpu())
        source_index = int(z_source_domain[row_idx].detach().cpu())
        rows.append({
            'split': split,
            'train_seed': train_seed,
            'domain_seed': domain_seed,
            'support_seed': support_seed,
            'query_seed': query_seed,
            'domain_index': domain_index,
            'query_index': row_idx % query_repeats,
            'arm': arm,
            'K': k,
            'z_source_domain_index': source_index,
            'final_goal_error': float(final_error[row_idx].cpu()),
            'success': bool(success[row_idx].cpu()),
            'first_query_action_mse': float(first_mse[row_idx].cpu()),
            'query_action_mse': float(query_mse[row_idx].cpu()),
            'delta_norm_last': float(delta_norm[row_idx].cpu()),
            'latent_delta_norm': float(latent_delta[source_index].detach().cpu()) if arm in ('ours', 'shuffle_z') else 0.0,
        })
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['arm']}:K{row['K']}", []).append(row)
    return {
        key: {
            'n': len(vals),
            'mean_final_goal_error': sum(float(v['final_goal_error']) for v in vals) / len(vals),
            'success_rate': sum(1.0 if v['success'] else 0.0 for v in vals) / len(vals),
            'mean_first_query_action_mse': sum(float(v['first_query_action_mse']) for v in vals) / len(vals),
            'mean_query_action_mse': sum(float(v['query_action_mse']) for v in vals) / len(vals),
        }
        for key, vals in sorted(grouped.items())
    }


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg = Toy2DConfig()
    env = Toy2DVectorEnv(cfg, device=device)
    system = build_system(cfg, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    system.load_state_dict(ckpt['system'])
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    before_hash = tensor_state_hash(system)
    domain = sample_toy2d_domains(args.num_domains, args.domain_seed, cfg, device=device)
    loss_cfg = AdaptationLossConfig(lambda_cos=args.lambda_cos, lambda_z=args.lambda_z)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    support_seed = args.domain_seed + 1000
    shuffle_perm = fixed_derangement(args.num_domains, args.shuffle_seed, device)
    shuffle_map = {int(i): int(shuffle_perm[i].detach().cpu()) for i in range(args.num_domains)}
    config = {
        'split': args.split,
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': file_hash(args.checkpoint),
        'train_seed': args.train_seed,
        'domain_seed': args.domain_seed,
        'support_seed': support_seed,
        'query_seed': args.query_seed,
        'shuffle_seed': args.shuffle_seed,
        'shuffle_map_domain_to_z_source': shuffle_map,
        'num_domains': args.num_domains,
        'query_repeats': args.query_repeats,
        'k_values': args.k_values,
        'query_horizon': args.query_horizon,
        'arms': ['base', 'frozen_z', 'ours', 'shuffle_z'],
        'notes': 'same support prefix for every K; params frozen during eval; only per-domain z is updated in support; query is no_grad and z-frozen',
    }
    (outdir / 'eval_config.json').write_text(json.dumps(config, indent=2, sort_keys=True))
    query_domain = repeat_domain(domain, args.query_repeats)
    query_domain_ids = query_domain.domain_id
    rows: List[Dict[str, object]] = []
    for k in args.k_values:
        support_obs = env.reset(args.num_domains, support_seed, domain=domain)
        with torch.enable_grad():
            z, _support_tail, support_metrics = support_adapt(
                system,
                env,
                domain,
                support_obs,
                k=k,
                loss_config=loss_cfg,
                create_graph=False,
            )
        z = z.detach()
        latent_delta = torch.linalg.norm(z - system.z0.detach().unsqueeze(0), dim=-1)
        z_query = z.repeat_interleave(args.query_repeats, dim=0)
        frozen_query = system.z0.detach().unsqueeze(0).expand(args.num_domains, -1).clone().repeat_interleave(args.query_repeats, dim=0)
        shuffle_query = z[shuffle_perm].repeat_interleave(args.query_repeats, dim=0)
        source_identity = torch.arange(args.num_domains, device=device).repeat_interleave(args.query_repeats)
        source_shuffle = shuffle_perm.repeat_interleave(args.query_repeats)
        arm_specs = [
            ('base', z_query, source_identity),
            ('frozen_z', frozen_query, source_identity),
            ('ours', z_query, source_identity),
            ('shuffle_z', shuffle_query, source_shuffle),
        ]
        for arm, z_arm, source_domain in arm_specs:
            query_obs = env.reset(args.num_domains * args.query_repeats, args.query_seed, domain=query_domain)
            arm_rows = rollout_query(
                system,
                env,
                query_domain,
                query_obs,
                z_arm,
                horizon=args.query_horizon,
                arm=arm,
                k=k,
                train_seed=args.train_seed,
                split=args.split,
                domain_seed=args.domain_seed,
                support_seed=support_seed,
                query_seed=args.query_seed,
                query_repeats=args.query_repeats,
                z_source_domain=source_domain,
                latent_delta=latent_delta,
            )
            for row in arm_rows:
                row['support_loss'] = float(support_metrics['support_loss'])
            rows.extend(arm_rows)
    after_hash = tensor_state_hash(system)
    raw_path = outdir / 'eval_raw.jsonl'
    with raw_path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    summary = {
        'raw_path': str(raw_path),
        'summary': summarize(rows),
        'config': config,
        'freeze_audit': {
            'parameter_hash_before_eval': before_hash,
            'parameter_hash_after_eval': after_hash,
            'unchanged': before_hash == after_hash,
        },
    }
    (outdir / 'eval_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--split', default='dev')
    parser.add_argument('--domain-seed', type=int, default=31000)
    parser.add_argument('--query-seed', type=int, default=41000)
    parser.add_argument('--shuffle-seed', type=int, default=51000)
    parser.add_argument('--num-domains', type=int, default=128)
    parser.add_argument('--query-repeats', type=int, default=8)
    parser.add_argument('--query-horizon', type=int, default=10)
    parser.add_argument('--k-values', type=int, nargs='+', default=[0, 1, 2, 3, 5])
    parser.add_argument('--lambda-cos', type=float, default=0.1)
    parser.add_argument('--lambda-z', type=float, default=1.0e-4)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
