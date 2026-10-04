"""Native demonstration and training-only common-support collection."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p28_common import (
    R, TASKS, BLOCK, HORIZON, SUPPORT, TaskEnv, Visual, task_root,
    transform, save_json, digest, load_base)


def collect_block(task, seeds, matrices, horizon, mode, visual, base=None, labels=True):
    envs = [TaskEnv(task, m) for m in matrices]
    n = len(envs)
    positions = [np.stack([e.reset(s) for e, s in zip(envs, seeds)])]
    features, actions, teachers, states, rgbs = [], [], [], [], []
    previous = torch.zeros(n, 4, device='cuda')
    began = time.time()
    for t in range(horizon+1):
        images = np.stack([e.rgb() for e in envs])
        v = visual.encode(images)
        features.append(v.cpu().half() if visual.pca is None else v.cpu().float())
        states.append({k: np.stack([e.state()[k] for e in envs])
                       for k in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat')})
        if t in (0, horizon//2, horizon):
            rgbs.append(images[:2])
        if t == horizon:
            break
        if labels:
            teacher = np.stack([e.teacher() for e in envs])
            teachers.append(teacher)
        if mode == 'expert':
            action = teacher.copy()
            # Small deterministic cartesian noise only; original labels retained.
            for i, s in enumerate(seeds):
                action[i, :3] += np.random.default_rng(int(s)*1000+t).uniform(-.02, .02, 3)
            action = np.clip(action, -1, 1)
        else:
            with torch.no_grad():
                action = base(v, torch.as_tensor(positions[-1], device='cuda'), previous).cpu().numpy()
        positions.append(np.stack([e.step(a) for e, a in zip(envs, action)]))
        actions.append(action)
        previous = torch.as_tensor(action, device='cuda')
    metrics = [e.metrics for e in envs]
    for e in envs:
        e.close()
    values = {'visual': torch.stack(features, 1),
              'proprio': torch.tensor(np.stack(positions, 1), dtype=torch.float32),
              'actions': torch.tensor(np.stack(actions, 1), dtype=torch.float32),
              'seeds': [int(s) for s in seeds], 'mode': mode,
              'states': {k: torch.tensor(np.stack([s[k] for s in states], 1)) for k in states[0]},
              'audit_rgb': torch.tensor(np.stack(rgbs, 1)), 'seconds': time.time()-began}
    if labels:
        values['teacher'] = torch.tensor(np.stack(teachers, 1), dtype=torch.float32)
    return values, metrics


def qualify(task):
    rows = []
    for mode in ('native', 'response_ood'):
        for i in range(16):
            seed = 2801000+10000*TASKS.index(task)+i
            e = TaskEnv(task, transform(seed+100000, mode), render=False)
            e.reset(seed)
            for _ in range(HORIZON):
                e.step(e.teacher())
            row = {'mode': mode, 'seed': seed,
                   'success_any': any(x['success'] for x in e.metrics),
                   'success_final': bool(e.metrics[-1]['success'])}
            rows.append(row); e.close()
    passed = all(sum(x['success_any'] for x in rows if x['mode'] == m) >= 14
                 for m in ('native', 'response_ood'))
    save_json(task_root(task)/'qualification/TEACHER.json',
              {'passed': passed, 'rows': rows, 'source': 'unmodified upstream expert with training-only inverse action wrapper'})
    print(task, 'teacher qualification', passed, flush=True)
    return passed


def collect(task, stage, dino):
    torch.set_num_threads(2)
    out = task_root(task)/'data'/stage
    out.mkdir(parents=True, exist_ok=False)
    pca = None if stage == 'demos' else torch.load(task_root(task)/'data/PCA.pt', weights_only=True)
    visual = Visual(dino, pca)
    base = None
    if stage == 'dagger':
        from prove.fast_adapter.metaworld_adapter import RGBSkill
        base = RGBSkill().cuda()
        base.load_state_dict(torch.load(task_root(task)/'models/base_bc.pt', weights_only=True)['state'])
        base.eval().requires_grad_(False)
    elif stage == 'meta':
        base = load_base(task)
    count = 128 if stage == 'demos' else 64
    task_offset = TASKS.index(task)*10000
    origin = {'demos': 2810000, 'dagger': 2820000, 'meta': 2830000}[stage]+task_offset
    roles = ('support', 'expert', 'base') if stage == 'meta' else (stage,)
    receipt = []
    for role_index, role in enumerate(roles):
        for offset in range(0, count, BLOCK):
            indices = np.arange(offset, min(count, offset+BLOCK))
            seeds = origin+100000*role_index+indices
            matrices = [transform(origin+300000+int(i), 'train' if stage == 'meta' else 'native') for i in indices]
            horizon = SUPPORT if role == 'support' else HORIZON
            mode = 'expert' if role in ('demos', 'expert') else 'base'
            blob, metrics = collect_block(task, seeds, matrices, horizon, mode, visual, base,
                                          labels=role != 'support')
            target = out/f'{role}_{offset:03d}.pt'
            torch.save(blob, target)
            # Metric reporting happens after runtime records are sealed on disk.
            successes = [any(x['success'] for x in xs) for xs in metrics]
            rec = {'file': str(target), 'sha256': digest(target), 'role': role,
                   'offset': offset, 'n': len(indices), 'successes': int(sum(successes)),
                   'seconds': blob['seconds']}
            receipt.append(rec)
            save_json(out/f'{role}_{offset:03d}_EVALUATOR.json', {'source_sha256': rec['sha256'],
                       'matrices': [m.tolist() for m in matrices], 'metrics': metrics})
            print(task, stage, role, offset, f'{sum(successes)}/{len(indices)}',
                  f"{blob['seconds']:.1f}s", flush=True)
    save_json(out/'COMPLETE.json', {'stage': stage, 'task': task, 'files': receipt,
                                  'test_data_used': False})


def merge_demos(task):
    torch.set_num_threads(4)
    root = task_root(task)/'data'
    paths = sorted((root/'demos').glob('demos_*.pt'))
    sample = torch.cat([torch.load(p, weights_only=True)['visual'][:, ::8].flatten(0, 1)
                        for p in paths]).float().cuda()
    mean = sample.mean(0); centered = sample-mean
    torch.manual_seed(28028)
    _, sv, basis = torch.pca_lowrank(centered, q=80, center=False, niter=4)
    basis = basis[:, :64]
    sd = (sv[:64].square()/(len(sample)-1)).clamp_min(1e-6).sqrt()
    pca = {'mean': mean.cpu(), 'basis': basis.cpu(), 'sd': sd.cpu(),
           'source': 'native training demonstrations only, every8th frame',
           'retained_variance': float(sv[:64].square().sum()/centered.square().sum())}
    torch.save(pca, root/'PCA.pt')
    del sample, centered
    banks = []
    for path in paths:
        b = torch.load(path, weights_only=True)
        projected = []
        for v in b['visual'].split(8):
            projected.append(((v.float().cuda()-mean)@basis/sd).cpu())
        banks.append({'visual': torch.cat(projected),
                      **{k: b[k] for k in ('proprio', 'actions', 'teacher')}})
    torch.save({k: torch.cat([b[k] for b in banks]) for k in banks[0]}, root/'DEMOS.pt')
    save_json(root/'PCA_SEAL.json', {'pca_sha256': digest(root/'PCA.pt'),
              'demonstrations_sha256': digest(root/'DEMOS.pt'),
              'sources': [{'path': str(p), 'sha256': digest(p)} for p in paths],
              'pca_fit': pca['source'], 'retained_variance': pca['retained_variance']})
    print(task, 'PCA complete', pca['retained_variance'], flush=True)


def load_stage(task, stage, role=None):
    root = task_root(task)/'data'
    if stage == 'demos':
        return torch.load(root/'DEMOS.pt', weights_only=True)
    paths = sorted((root/stage).glob(f'{role or stage}_[0-9]*.pt'))
    assert paths
    banks = [torch.load(p, weights_only=True) for p in paths]
    keys = ['visual', 'proprio', 'actions']+(['teacher'] if 'teacher' in banks[0] else [])
    return {k: torch.cat([b[k] for b in banks]) for k in keys}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['qualify', 'collect', 'merge'])
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--stage', choices=['demos', 'dagger', 'meta']); p.add_argument('--dino', default='assets/dinov3')
    a = p.parse_args()
    if a.mode == 'qualify':
        raise SystemExit(0 if qualify(a.task) else 2)
    elif a.mode == 'merge':
        merge_demos(a.task)
    else:
        collect(a.task, a.stage, a.dino)
