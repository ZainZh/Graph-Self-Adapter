"""Matched frozen-policy rollouts, latent-source interventions, no outcome selection."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p28_common import (R, TASKS, SEEDS, BLOCK, HORIZON, SUPPORT,
    TaskEnv, Visual, task_root, transform, save_json, digest, load_base, load_adapter, weight_hash)
from tools.fast_adapter.p28_data import collect_block
from tools.fast_adapter.p28_train import support_inputs

EVAL_STRATA = ('native', 'response_id', 'response_ood')


def execute(task, seeds, matrices, visual, base, path, net=None, latent=None):
    envs = [TaskEnv(task, m) for m in matrices]
    positions = [np.stack([e.reset(s) for e, s in zip(envs, seeds)])]
    features, actions, bases, states, audit_rgb = [], [], [], [], []
    previous = torch.zeros(len(envs), 4, device='cuda')
    for t in range(HORIZON+1):
        images = np.stack([e.rgb() for e in envs])
        v = visual.encode(images)
        features.append(v.cpu().float())
        states.append({k: np.stack([e.state()[k] for e in envs]) for k in
                       ('qpos', 'qvel', 'mocap_pos', 'mocap_quat')})
        if t in (0, HORIZON//2, HORIZON):
            audit_rgb.append(images[:2])
        if t == HORIZON:
            break
        with torch.no_grad():
            p = torch.as_tensor(positions[-1], device='cuda')
            ba = base(v, p, previous)
            a = ba if net is None else net.action(v, p, previous, ba, latent)
            if net is not None:
                k0 = net.action(v, p, previous, ba, net.initial(len(v)))
                assert torch.equal(k0, ba), 'K0 identity failure'
            assert torch.isfinite(a).all()
        bases.append(ba.cpu()); actions.append(a.cpu())
        positions.append(np.stack([e.step(x) for e, x in zip(envs, a.cpu().numpy())]))
        previous = a
    blob = {'seeds': [int(s) for s in seeds],
            'visual': torch.stack(features, 1),
            'proprio': torch.tensor(np.stack(positions, 1), dtype=torch.float32),
            'actions': torch.stack(actions, 1), 'base': torch.stack(bases, 1),
            'states': {k: torch.tensor(np.stack([s[k] for s in states], 1)) for k in states[0]},
            'audit_rgb': torch.tensor(np.stack(audit_rgb, 1)),
            'latent': None if latent is None else latent.cpu(), 'updates_in_query': 0}
    torch.save(blob, path)
    record_hash = digest(path)
    # Evaluator-only metrics are exported after the policy record is sealed.
    metrics = [e.metrics for e in envs]
    for e in envs:
        e.close()
    result = {'file': str(path), 'sha256': record_hash,
              'success': [bool(any(x['success'] for x in xs)) for xs in metrics],
              'terminal_success': [bool(xs[-1]['success']) for xs in metrics],
              'first_success': [next((i+1 for i, x in enumerate(xs) if x['success']), None) for xs in metrics],
              'matrices': [m.tolist() for m in matrices], 'metrics': metrics}
    save_json(path.with_suffix('.json'), result)
    return blob['states']['qpos'][:, 0], result


def base_qualification(task, dino):
    torch.set_num_threads(2)
    root = task_root(task); out = root/'qualification'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'BASE.json').exists()
    base = load_base(task)
    pca = torch.load(root/'data/PCA.pt', weights_only=True)
    visual = Visual(dino, pca)
    seeds = np.arange(16)+3900000+100000*TASKS.index(task)+5000000
    _, result = execute(task, seeds, [np.eye(3)]*16, visual, base, out/'BASE_RAW.pt')
    n = sum(result['success'])
    save_json(out/'BASE.json', {'passed': n >= 8, 'successes': n, 'episodes': 16,
              'success': result['success'], 'terminal_success': result['terminal_success'],
              'source_sha256': result['sha256'], 'base_sha256': digest(root/'models/base_final.pt')})
    print('BASE_QUALIFICATION', task, n, '/16', flush=True)
    return n >= 8


def evaluate(task, stratum, dino):
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    root = task_root(task); out = root/'evaluation'/stratum
    out.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((root/'MODEL_FREEZE.json').read_text())
    for b in freeze['bindings']:
        assert digest(b['path']) == b['sha256'], b['path']
    base = load_base(task)
    pca = torch.load(root/'data/PCA.pt', weights_only=True)
    visual = Visual(dino, pca)
    nets = {(s, v): load_adapter(task, s, v) for s in SEEDS for v in ('fast', 'history')}
    before = {'base': weight_hash(base), **{f'{s}_{v}': weight_hash(n[0]) for (s, v), n in nets.items()}}
    origin = 4900000+100000*TASKS.index(task)+20000*EVAL_STRATA.index(stratum)
    matrix_mode = 'train' if stratum == 'response_id' else stratum
    rows = []
    began = time.time()
    for offset in range(0, 64, BLOCK):
        ids = origin+np.arange(offset, offset+BLOCK)
        matrices = [transform(int(s)+1000000, matrix_mode) for s in ids]
        donor_matrices = [transform(int(s)+3000000, matrix_mode) for s in ids]
        supports = {}
        support_seals = []
        for source, ms, delta in [('correct', matrices, 2000000), ('donor', donor_matrices, 4000000)]:
            blob, metrics = collect_block(task, ids+delta, ms, SUPPORT, 'base', visual, base, labels=False)
            p = out/f'SUPPORT_{source}_{offset:03d}.pt'; torch.save(blob, p)
            support_seals.append({'path': str(p), 'sha256': digest(p)})
            supports[source] = support_inputs({k: blob[k].cuda() for k in ('visual', 'proprio', 'actions')})
        latents = {}
        for s in SEEDS:
            fast, fblob = nets[(s, 'fast')]
            history, _ = nets[(s, 'history')]
            latents[f'fast_correct_seed{s}'] = fast.adapt(supports['correct'], 3)[0]
            latents[f'fast_donor_seed{s}'] = fast.adapt(supports['donor'], 3)[0]
            latents[f'fast_mean_seed{s}'] = fblob['mean_z'].cuda()[None].expand(BLOCK, -1)
            latents[f'history_seed{s}'] = history.adapt(supports['correct'], 3)[0]
        lp = out/f'LATENTS_{offset:03d}.pt'
        torch.save({k: v.cpu() for k, v in latents.items()}, lp)
        save_json(out/f'QUERY_START_SEAL_{offset:03d}.json', {'query_not_started': True,
                  'support_bindings': support_seals, 'latents': {'path': str(lp), 'sha256': digest(lp)},
                  'model_freeze_sha256': digest(root/'MODEL_FREEZE.json')})
        initial, result = execute(task, ids+5000000, matrices, visual, base, out/f'base_{offset:03d}.pt')
        rows.append({'arm': 'base', 'offset': offset, **{k: v for k, v in result.items() if k not in ('metrics', 'matrices')}})
        for seed in SEEDS:
            for variant in ('fast_correct', 'fast_mean', 'fast_donor', 'history'):
                name = f'{variant}_seed{seed}'
                net = nets[(seed, 'history' if variant == 'history' else 'fast')][0]
                initial_arm, result = execute(task, ids+5000000, matrices, visual, base,
                    out/f'{name}_{offset:03d}.pt', net, latents[name])
                assert torch.equal(initial, initial_arm), 'unmatched query reset'
                rows.append({'arm': name, 'offset': offset, **{k: v for k, v in result.items() if k not in ('metrics', 'matrices')}})
        print(task, stratum, 'complete_conditions', offset+BLOCK, 'seconds', time.time()-began, flush=True)
    after = {'base': weight_hash(base), **{f'{s}_{v}': weight_hash(n[0]) for (s, v), n in nets.items()}}
    assert before == after
    save_json(out/'RESULT.json', {'task': task, 'stratum': stratum, 'conditions': 64,
              'query_episodes': 64*9, 'support_episodes': 128,
              'seconds': time.time()-began, 'networks_unchanged': True,
              'weights': before, 'initial_states_matched': True, 'rows': rows})
    print('EVAL_COMPLETE', task, stratum, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['qualify', 'evaluate'])
    p.add_argument('--task', required=True, choices=TASKS); p.add_argument('--stratum', choices=EVAL_STRATA)
    p.add_argument('--dino', default='assets/dinov3')
    a = p.parse_args()
    if a.mode == 'qualify':
        # Qualification failures are scientific states, not process failures.
        base_qualification(a.task, a.dino)
    else:
        evaluate(a.task, a.stratum, a.dino)
