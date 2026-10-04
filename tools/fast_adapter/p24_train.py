"""50k full meta updates and 50k matched History updates, with live data refreshes."""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import time
import torch
from prove.fast_adapter.skill_adapter import SkillAdapter
from prove.fast_adapter.insert_adapter import support_actions
from tools.fast_adapter.p24_common import R, PARENT, load_base, weight_hash
from tools.fast_adapter.p18_run import save_json, digest


def load_bank(path):
    bank = torch.load(path, weights_only=True)
    keys = ['support_visual', 'query_visual', 'proprio', 'expert_label', 'query_action']
    out = {k: bank[k].cuda() for k in keys}
    out['support_action'] = bank.get('support_action', support_actions(len(bank['proprio']))).cuda()
    return out


def train(seed):
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    protocol = json.loads((R / 'PROTOCOL.json').read_text())
    total = protocol['training']['updates_per_model']
    out = R / 'models' / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    base = load_base(); base_hash = weight_hash(base)
    meta = SkillAdapter().cuda()
    history = SkillAdapter(history=True).cuda()
    history.load_state_dict(meta.state_dict(), strict=False)
    nets = {'meta': meta, 'history': history}
    opts = {name: torch.optim.Adam(net.parameters(), lr=.0003) for name, net in nets.items()}
    bank = load_bank(PARENT / 'bank/TRAIN.pt')
    bank_sources = [{'path': str(PARENT / 'bank/TRAIN.pt'), 'sha256': digest(PARENT / 'bank/TRAIN.pt')}]
    rng = torch.Generator().manual_seed(24000 + seed)
    began = time.time(); k_counts = {str(k): 0 for k in (1, 2, 3, 5)}
    log = (out / 'TRAIN.jsonl').open('x')
    for step in range(1, total + 1):
        ids = torch.randint(len(bank['proprio']), (128,), generator=rng).cuda()
        times = torch.randint(48, (128, 4), generator=rng).cuda()
        k = (1, 2, 3, 5)[int(torch.randint(4, (), generator=rng))]
        k_counts[str(k)] += 1
        bi = ids[:, None].expand(-1, 4)
        sv = bank['support_visual'][ids, :k+1]
        sp = bank['proprio'][ids, :k+1]
        sa = bank['support_action'][ids, :k]
        v = bank['query_visual'][bi, times]
        p = bank['proprio'][:, 6:][bi, times]
        executed = bank['query_action'][bi, times]
        previous = bank['query_action'][bi, (times-1).clamp_min(0)] * (times > 0).unsqueeze(-1)
        nv = bank['query_visual'][bi, times+1]
        np = bank['proprio'][:, 6:][bi, times+1]
        with torch.no_grad(): ba = base(v, p)
        row = {'step': step, 'support_transitions': k, 'inner_updates': k, 'training_episode_records': len(bank['proprio'])}
        for name, net in nets.items():
            lr = .00003 + .5 * (.0003-.00003) * (1 + math.cos(math.pi*(step-1)/total))
            for group in opts[name].param_groups: group['lr'] = lr
            z = net.latent(sv, sp, sa, meta=name == 'meta')
            zz = z[:, None, :].expand(-1, 4, -1)
            u = net.encode(v, p, previous)
            act = net.action_u(u, ba, zz)
            action_loss = ((act-bank['expert_label'][bi, times])/.35).square().mean()
            target = net.encode(nv, np, executed).detach()
            prediction_loss = (net.predict(u, executed, zz)-target).square().mean()
            loss = action_loss + .1 * prediction_loss
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite {name} loss at {step}')
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            if not torch.isfinite(grad_norm): raise RuntimeError(f'nonfinite {name} gradient at {step}')
            opts[name].step()
            if step == 1 or step % 1000 == 0:
                row[name] = {'action_loss': float(action_loss.detach()),
                             'prediction_loss': float(prediction_loss.detach()),
                             'alpha': float(net.alpha.detach()),
                             'z_rms_from_prior': float((z.detach()-net.initial(len(z)).detach()).square().mean().sqrt()),
                             'representation_std': float(u.detach().flatten(0, 1).std(0).mean()),
                             'outer_gradient_norm': float(grad_norm)}
        if step == 1 or step % 1000 == 0:
            row['seconds'] = time.time()-began; row['k_counts'] = dict(k_counts)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
        if step in protocol['training']['refresh_after_updates']:
            checkpoint = out / f'meta_step{step}.pt'
            torch.save({'state': meta.state_dict(), 'updates': step, 'seed': seed}, checkpoint)
            command = [sys.executable, '-m', 'tools.fast_adapter.p24_rollout', 'collect',
                       '--seed', str(seed), '--checkpoint', str(checkpoint), '--round', str(step),
                       '--dino', 'assets/dinov3']
            with (out / f'collect_step{step}.log').open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            packet = R / 'training_refresh' / f'seed{seed}' / f'step{step}/BANK.pt'
            fresh = load_bank(packet)
            bank = {key: torch.cat((bank[key], fresh[key])) for key in bank}
            bank_sources.append({'path': str(packet), 'sha256': digest(packet)})
    assert weight_hash(base) == base_hash
    for name, net in nets.items():
        torch.save({'state': net.state_dict(), 'seed': seed, 'updates': total,
                    'kind': name, 'k_counts': k_counts}, out / f'{name}.pt')
    log.close()
    save_json(out / 'COMPLETE.json', {'seed': seed, 'updates_per_model': total,
        'meta_updates': total, 'history_updates': total, 'no_history_warmup': True,
        'k_counts': k_counts, 'base_weight_hash': base_hash, 'base_weights_unchanged': True,
        'bank_sources': bank_sources, 'seconds': time.time()-began,
        'models': {name: {'sha256': digest(out/f'{name}.pt'),
                         'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad)}
                   for name, net in nets.items()}})
    print('TRAIN_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, required=True)
    train(p.parse_args().seed)
