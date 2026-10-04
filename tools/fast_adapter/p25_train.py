"""Matched small retraining on existing complete training trajectories only."""
import argparse
import json
import math
import time
import torch
from prove.fast_adapter.feedback_adapter import FeedbackAdapter
from tools.fast_adapter.p25_common import R, VARIANTS, make_bank, select_feedback, transitions, load_base, weight_hash
from tools.fast_adapter.p18_run import digest, save_json


def main(seed):
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False; torch.manual_seed(seed)
    out = R/'models'/f'seed{seed}'; out.mkdir(parents=True, exist_ok=False)
    bank, paths = make_bank(); bank = {k: v.cuda() for k, v in bank.items()}
    phase = select_feedback(bank['proprio'].flatten(0, 1), bank['actions'].flatten(0, 1)).reshape(6, 384, 3)
    base = load_base(); base_hash = weight_hash(base)
    nets = {name: FeedbackAdapter(name).cuda() for name in VARIANTS}
    reference = nets['original'].state_dict()
    for name, net in nets.items():
        if name != 'original':
            state = net.state_dict()
            state.update({k: v.clone() for k, v in reference.items() if k in state and state[k].shape == v.shape})
            net.load_state_dict(state)
    optimizers = {name: torch.optim.Adam(net.parameters(), lr=.0003) for name, net in nets.items()}
    generator = torch.Generator().manual_seed(25000+seed)
    total = json.loads((R/'PROTOCOL.json').read_text())['training']['updates']
    began = time.time()
    with (out/'TRAIN.jsonl').open('x') as stream:
        for step in range(1, total+1):
            d = torch.randint(384, (64,), generator=generator).cuda()
            sr = torch.randint(6, (64,), generator=generator).cuda()
            qr = (sr + torch.randint(1, 6, (64,), generator=generator).cuda()) % 6
            t = torch.randint(48, (64, 4), generator=generator).cuda()
            support = transitions(bank['visual'][sr, d], bank['proprio'][sr, d], bank['actions'][sr, d], phase[sr, d])
            q, dd = qr[:, None], d[:, None]
            v, p = bank['visual'][q, dd, t], bank['proprio'][q, dd, t]
            nv, np_ = bank['visual'][q, dd, t+1], bank['proprio'][q, dd, t+1]
            executed = bank['actions'][q, dd, t]
            previous = bank['actions'][q, dd, (t-1).clamp_min(0)] * (t > 0).unsqueeze(-1)
            with torch.no_grad(): ba = base(v, p)
            row = {'step': step, 'seconds': time.time()-began}
            for name, net in nets.items():
                optimizer = optimizers[name]
                for group in optimizer.param_groups:
                    group['lr'] = .00003 + .5*(.0003-.00003)*(1+math.cos(math.pi*(step-1)/total))
                path, _ = net.feedback_path(support, meta=name != 'history')
                z = path[-1][:, None].expand(-1, 4, -1)
                u = net.encode(v, p, previous)
                action = net.action_u(u, ba, z)
                label = bank['expert'][q, dd, t]
                action_loss = ((action-label)/.35).square().mean()
                next_u = net.encode(nv, np_, executed)
                target = net.signal_target(next_u, p, np_)
                prediction_loss = (net.signal_prediction(u, executed, z)-target).square().mean()
                loss = action_loss + .1*prediction_loss
                assert torch.isfinite(loss), (name, step)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
                assert torch.isfinite(norm), (name, step, 'gradient')
                optimizer.step()
                if step == 1 or step % 500 == 0:
                    row[name] = {'action_loss': float(action_loss.detach()), 'prediction_loss': float(prediction_loss.detach()),
                        'alpha': float(net.alpha.detach()), 'z_delta_rms': float((path[-1]-path[0]).detach().square().mean().sqrt())}
            if step == 1 or step % 500 == 0:
                stream.write(json.dumps(row)+'\n'); stream.flush(); print(json.dumps(row), flush=True)
    assert weight_hash(base) == base_hash
    for name, net in nets.items():
        net.eval().requires_grad_(False)
        latent = []
        for tr in range(6):
            for first in range(0, 384, 64):
                sl = slice(first, first+64)
                feedback = transitions(bank['visual'][tr, sl], bank['proprio'][tr, sl], bank['actions'][tr, sl], phase[tr, sl])
                path, _ = net.feedback_path(feedback)
                latent.append(path[-1].detach())
        mean_z = torch.cat(latent).mean(0).cpu()
        torch.save({'state': net.state_dict(), 'training_mean_z': mean_z, 'updates': total, 'seed': seed,
                    'mean_z_source': 'All 2304 old training query trajectories, three selected transitions each; no development inputs.'}, out/f'{name}.pt')
    save_json(out/'COMPLETE.json', {'seed': seed, 'updates_each': total, 'variants': list(VARIANTS),
        'base_weights_unchanged': True, 'support_query_trajectories_distinct': True,
        'training_conditions': 384, 'reused_query_trajectories': 2304,
        'source_banks': [{'path': str(p), 'sha256': digest(p)} for p in paths],
        'models': {name: digest(out/f'{name}.pt') for name in VARIANTS}, 'seconds': time.time()-began})
    print('TRAIN_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, required=True)
    main(p.parse_args().seed)
