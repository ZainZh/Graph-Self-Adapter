"""Fixed-budget per-task visual skill fitting and latent meta-training."""
import argparse
import json
import math
import time
import torch
from prove.fast_adapter.metaworld_adapter import RGBSkill, RobotAdapter
from tools.fast_adapter.p28_common import task_root, TASKS, SEEDS, load_base, save_json, digest, weight_hash
from tools.fast_adapter.p28_data import load_stage


def gpu(bank):
    return {k: v.cuda() for k, v in bank.items()}


def previous(bank):
    return torch.cat((torch.zeros_like(bank['actions'][:, :1]), bank['actions']), 1)


def support_inputs(bank):
    return {'visual': bank['visual'][:, :-1], 'proprio': bank['proprio'][:, :-1],
            'next_proprio': bank['proprio'][:, 1:], 'previous': previous(bank)[:, :-1],
            'actions': bank['actions']}


def base_train(task, stage):
    torch.set_num_threads(2); torch.manual_seed(2811)
    root = task_root(task); out = root/'models'; out.mkdir(parents=True, exist_ok=True)
    bank = gpu(load_stage(task, 'demos'))
    model = RGBSkill().cuda()
    if stage == 'dagger':
        model.load_state_dict(torch.load(out/'base_bc.pt', weights_only=True)['state'])
        new = gpu(load_stage(task, 'dagger'))
        # Explicit50/50 mixture of demonstrations and on-policy training states.
        bank = {k: torch.cat((bank[k], new[k], new[k]), 0) for k in bank}
    n, horizon = bank['actions'].shape[:2]
    prev = previous(bank)
    total = 12000 if stage == 'bc' else 6000
    optimizer = torch.optim.Adam(model.parameters(), lr=.0006)
    began = time.time()
    with (out/f'BASE_{stage}.jsonl').open('x') as log:
        for step in range(1, total+1):
            d = torch.randint(n, (512,), device='cuda')
            t = torch.randint(horizon, (512,), device='cuda')
            action = model(bank['visual'][d, t], bank['proprio'][d, t], prev[d, t])
            loss = (action-bank['teacher'][d, t]).square().mean()
            for group in optimizer.param_groups:
                group['lr'] = .00003+.5*(.0006-.00003)*(1+math.cos(math.pi*step/total))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step == 1 or step % 1000 == 0:
                row = {'step': step, 'loss': float(loss.detach()), 'seconds': time.time()-began}
                log.write(json.dumps(row)+'\n'); log.flush(); print(task, stage, row, flush=True)
    target = out/('base_bc.pt' if stage == 'bc' else 'base_final.pt')
    torch.save({'state': model.cpu().state_dict(), 'seed': 2811, 'updates_this_stage': total,
                'total_updates': total if stage == 'bc' else 18000,
                'inputs': ['frozen_LiLa_PCA64', 'EEF_xyz', 'gripper_aperture', 'previous_requested_action']}, target)
    save_json(out/f'BASE_{stage}_COMPLETE.json', {'path': str(target), 'sha256': digest(target),
              'updates': total, 'training_episodes': n, 'seconds': time.time()-began,
              'parameter_count': sum(p.numel() for p in model.parameters())})


def adapt_train(task, seed):
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    root = task_root(task); out = root/'models'/f'seed{seed}'; out.mkdir(parents=True, exist_ok=False)
    support = support_inputs(gpu(load_stage(task, 'meta', 'support')))
    queries = [gpu(load_stage(task, 'meta', role)) for role in ('expert', 'base')]
    for q in queries:
        q['previous'] = previous(q)
    base = load_base(task); base_hash = weight_hash(base)
    nets = {'fast': RobotAdapter().cuda(), 'history': RobotAdapter(history=True).cuda()}
    # Same shared-module initialization; History alone has its GRU components.
    shared = nets['fast'].state_dict(); state = nets['history'].state_dict()
    state.update({k: v.clone() for k, v in shared.items() if k in state})
    nets['history'].load_state_dict(state)
    opts = {k: torch.optim.Adam(n.parameters(), lr=.0003) for k, n in nets.items()}
    total = 16000; began = time.time(); n = len(support['actions'])
    with (out/'TRAIN.jsonl').open('x') as log:
        for step in range(1, total+1):
            ids = torch.randint(n, (32,), device='cuda')
            q = queries[step % 2]
            t = torch.randint(q['actions'].shape[1], (32, 8), device='cuda')
            d = ids[:, None]
            v, p, prev = q['visual'][d, t], q['proprio'][d, t], q['previous'][d, t]
            with torch.no_grad():
                ba = base(v, p, prev)
            inputs = {k: x[ids] for k, x in support.items()}
            ksteps = (1, 2, 3, 5)[int(torch.randint(4, ()).item())]
            row = {'step': step, 'inner_steps': ksteps, 'seconds': time.time()-began}
            for name, net in nets.items():
                z, _ = net.adapt(inputs, steps=ksteps, meta=True)
                zz = z[:, None].expand(-1, 8, -1)
                action = net.action(v, p, prev, ba, zz)
                act_loss = ((action-q['teacher'][d, t])/.5).square().mean()
                u = net.encode(v, p, prev)
                target = (q['proprio'][d, t+1, :3]-p[..., :3])/.01
                prediction = net.prediction(u, q['actions'][d, t], zz)
                pred_loss = (prediction-target).square().mean()
                loss = act_loss+.1*pred_loss
                assert torch.isfinite(loss), (task, seed, name, step)
                opt = opts[name]
                for group in opt.param_groups:
                    group['lr'] = .00003+.5*(.0003-.00003)*(1+math.cos(math.pi*step/total))
                opt.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
                assert torch.isfinite(norm), (task, seed, name, step)
                opt.step()
                if step == 1 or step % 1000 == 0:
                    row[name] = {'action_loss': float(act_loss.detach()), 'prediction_loss': float(pred_loss.detach()),
                                 'alpha': float(net.alpha.detach()),
                                 'z_delta_rms': float((z-net.z0).detach().square().mean().sqrt())}
            if step == 1 or step % 1000 == 0:
                log.write(json.dumps(row)+'\n'); log.flush(); print(task, seed, row, flush=True)
    assert weight_hash(base) == base_hash
    for name, net in nets.items():
        net.eval().requires_grad_(False)
        zs = []
        for offset in range(0, n, 16):
            z, _ = net.adapt({k: x[offset:offset+16] for k, x in support.items()}, 3)
            zs.append(z.cpu())
        torch.save({'state': net.cpu().state_dict(), 'seed': seed, 'updates': total,
                    'training_z': torch.cat(zs), 'mean_z': torch.cat(zs).mean(0),
                    'mean_source': 'All64 common frozen-base training support episodes,80 transitions,U3'},
                   out/f'{name}.pt')
    save_json(out/'COMPLETE.json', {'task': task, 'seed': seed, 'updates_each': total,
              'base_weights_unchanged': True, 'base_hash': base_hash,
              'support_query_resets_distinct': True,
              'models': {k: digest(out/f'{k}.pt') for k in nets}, 'seconds': time.time()-began})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['base', 'adapt'])
    p.add_argument('--task', required=True, choices=TASKS); p.add_argument('--stage', choices=['bc', 'dagger'])
    p.add_argument('--seed', type=int, choices=SEEDS)
    a = p.parse_args()
    if a.mode == 'base':
        base_train(a.task, a.stage)
    else:
        adapt_train(a.task, a.seed)
