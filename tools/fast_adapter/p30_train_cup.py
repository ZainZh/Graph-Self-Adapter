"""P30 bounded continuous online adaptation training on sealed P29 train banks.

This is a development trainer only. It never opens P29 final data and keeps the
physical transition order: each observed transition updates z before the next
action. The base policy is frozen; outer optimization updates adapter weights.
"""
import argparse, json, math, time
from pathlib import Path
import torch
from prove.fast_adapter.object_interaction_adapter import ObjectAdapter
from tools.fast_adapter.p29_common import task_root, digest, save_json, weight_hash
from tools.fast_adapter.p29_data import load_stage
from tools.fast_adapter.p29_train import gpu, previous
from prove.fast_adapter.online_latent import OnlineLatent

TASK = 'mug-handle-lift'
ROOT = Path('evaluations/fast_adapter_p30_online_v01')


def inputs(blob):
    prev = previous(blob)
    return {'visual': blob['visual'][:, :-1], 'next_visual': blob['visual'][:, 1:],
            'proprio': blob['proprio'][:, :-1], 'next_proprio': blob['proprio'][:, 1:],
            'previous': prev[:, :-1], 'actions': blob['actions']}


def train(seed, steps=240, batch=8, segment=20):
    torch.manual_seed(seed); torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    out = ROOT / 'models_continuous' / TASK / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    from tools.fast_adapter.p29_common import load_base
    base = load_base(TASK); base_hash = weight_hash(base)
    support = inputs(gpu(load_stage(TASK, 'meta', 'support')))
    expert = gpu(load_stage(TASK, 'meta', 'expert'))
    base_q = gpu(load_stage(TASK, 'meta', 'base'))
    expert['previous'] = previous(expert); base_q['previous'] = previous(base_q)
    net = ObjectAdapter().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    log_path = out / 'TRAIN.jsonl'; started = time.time()
    with log_path.open('x') as log:
        for step in range(1, steps + 1):
            ids = torch.randint(0, support['actions'].shape[0], (batch,), device='cuda')
            # Each sampled domain starts from z0 and is consumed in time order.
            ss = {k: x[ids] for k, x in support.items()}
            q = expert if step % 2 else base_q
            t0 = torch.randint(0, q['actions'].shape[1] - segment - 1, (batch,), device='cuda')
            learner = OnlineLatent(net, batch, window=8, mode='gradient', meta=True)
            total = 0.
            prev = torch.zeros(batch, 4, device='cuda')
            for j in range(segment):
                t = t0 + j
                v, p = q['visual'][ids, t], q['proprio'][ids, t]
                with torch.no_grad(): ba = base(v, p, prev)
                a = learner.act(v, p, prev, ba)
                total = total + ((a - q['teacher'][ids, t]) / .5).square().mean()
                # The recorded action and resulting next observation are the
                # actual transition used for self-supervised adaptation.
                learner.observe(q['visual'][ids, t + 1], q['proprio'][ids, t + 1],
                                q['actions'][ids, t])
                prev = q['actions'][ids, t].detach()
            loss = total / segment
            opt.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            if not bool(torch.isfinite(norm)): raise FloatingPointError('non-finite outer gradient')
            for group in opt.param_groups:
                group['lr'] = 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * step / steps))
            opt.step()
            if step == 1 or step % 20 == 0:
                row = {'step': step, 'loss': float(loss.detach()), 'grad_norm': float(norm),
                       'z_updates': learner.updates, 'z_delta_rms': float((learner.z - net.z0).detach().square().mean().sqrt()),
                       'seconds': time.time() - started}
                log.write(json.dumps(row) + '\n'); log.flush(); print(TASK, seed, row, flush=True)
    assert weight_hash(base) == base_hash
    net.eval().requires_grad_(False)
    path = out / 'gradient.pt'; torch.save({'state': net.cpu().state_dict(), 'seed': seed,
        'steps': steps, 'segment': segment, 'window': 8, 'base_hash': base_hash,
        'training_source': 'P29 meta train only; no P29 final', 'mode': 'continuous_gradient'}, path)
    save_json(out/'COMPLETE.json', {'status':'continuous_development_trained','model':str(path),
        'sha256':digest(path), 'seed':seed, 'steps':steps, 'segment':segment,
        'window':8, 'base_hash':base_hash, 'final_used':False, 'seconds':time.time()-started})
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=240); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--segment', type=int, default=20); a = p.parse_args()
    train(a.seed, a.steps, a.batch, a.segment)
