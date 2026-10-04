"""Live interaction collection and sealed queries for shared-prior adaptation."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
import mujoco
from prove.fast_adapter.geometry_insert import GeometryInsert, geometry, initial_position, domain
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL
from tools.fast_adapter.p22_data import RGBEncoder
from tools.fast_adapter.p24_common import R, PARENT, KS, load_base, load_adapter, weight_hash
from tools.fast_adapter.p18_run import save_json, digest


class SceneBatch:
    def __init__(self, configs, dynamics_seeds, encoder, pca):
        self.envs = [GeometryInsert(c) for c in configs]
        assert all(e.renderer._model is e.model for e in self.envs)
        for e, seed in zip(self.envs, dynamics_seeds): e.set_domain(*domain(seed))
        self.encoder, self.pca = encoder, pca
        self.last_rgb = None

    def observe(self):
        self.last_rgb = np.stack([e.rgb() for e in self.envs])
        raw = self.encoder.encode(self.last_rgb).half().float()
        v = (raw-self.pca['mean']) @ self.pca['basis'] / self.pca['sd']
        p = torch.tensor(np.stack([e.state() for e in self.envs]), device='cuda', dtype=torch.float32)
        return v, p

    def reset(self, seeds):
        for e, s in zip(self.envs, seeds): e.reset(initial_position(s))

    def step(self, action):
        for e, a in zip(self.envs, action): e.step(a)

    def close(self):
        for e in self.envs: e.close()


def visual_assets(dino):
    encoder = RGBEncoder(dino)
    pca = {k: v.cuda() if torch.is_tensor(v) else v
           for k, v in torch.load(PARENT/'bank/PCA.pt', weights_only=True).items()}
    return encoder, pca


def support(scene, net, base, start_seeds):
    """Causal online loop: act using z_k, then learn from that completed action."""
    scene.reset(start_seeds)
    v, p = scene.observe(); previous = torch.zeros_like(p)
    with torch.no_grad():
        u = net.encode(v, p, previous); z = net.initial(len(p)).detach().clone()
    vs, ps, actions, paths, losses, velocities = [v], [p], [], [z], [], []
    rgbs = [torch.tensor(scene.last_rgb[:2].copy())]
    for step in range(5):
        velocities.append(np.stack([e.data.qvel.copy() for e in scene.envs]))
        with torch.no_grad(): action = net.action_u(u, base(v, p), z)
        scene.step(action.cpu().numpy())
        nv, np_ = scene.observe()
        with torch.no_grad(): nu = net.encode(nv, np_, action)
        z, loss = net.update(u, action, nu, z)
        assert torch.isfinite(z).all() and torch.isfinite(loss).all()
        actions.append(action); vs.append(nv); ps.append(np_); paths.append(z); losses.append(loss)
        rgbs.append(torch.tensor(scene.last_rgb[:2].copy()))
        v, p, u = nv, np_, nu
    velocities.append(np.stack([e.data.qvel.copy() for e in scene.envs]))
    return {'visual': torch.stack(vs, 1).cpu(), 'proprio': torch.stack(ps, 1).cpu(),
            'actions': torch.stack(actions, 1).cpu(), 'meta_paths': torch.stack(paths, 1).cpu(),
            'per_transition_loss': torch.stack(losses, 1).cpu(),
            'velocities': torch.tensor(np.stack(velocities, 1)),
            'audit_rgb': torch.stack(rgbs), 'support_policy': 'current frozen meta adapter; one update per new transition'}


def query(scene, base, start_seeds, net=None, z=None, oracle=False, labels=False):
    scene.reset(start_seeds)
    n = len(scene.envs); previous = torch.zeros(n, 3, device='cuda')
    positions, velocities, actions, visual, bases, teacher, rgb = [], [], [], [], [], [], []
    for step in range(49):
        positions.append(np.stack([e.state() for e in scene.envs]))
        velocities.append(np.stack([e.data.qvel.copy() for e in scene.envs]))
        if not oracle:
            v, p = scene.observe(); visual.append(v.cpu().half())
            if step in (0, 24, 47): rgb.append(scene.last_rgb[:2].copy())
        if step == 48: break
        if oracle:
            act = np.stack([e.expert_action() for e in scene.envs])
        else:
            with torch.no_grad():
                ba = base(v, p); bases.append(ba.cpu())
                action = ba if net is None else net.action(v, p, previous, ba, z)
            act = action.cpu().numpy(); previous = action
        # Privileged labels never enter the action computation above.
        if labels: teacher.append(np.stack([e.expert_action() for e in scene.envs]))
        actions.append(act); scene.step(act)
    return {'positions': np.stack(positions, 1), 'velocities': np.stack(velocities, 1),
            'actions': np.stack(actions, 1),
            'visual': torch.stack(visual, 1) if visual else None,
            'base': torch.stack(bases, 1) if bases else None,
            'expert_label': np.stack(teacher, 1) if teacher else None,
            'audit_rgb': np.stack(rgb) if rgb else None, 'audit_rgb_steps': [0, 24, 47]}


def score(scene, raw):
    success, checks = [], []
    for i, e in enumerate(scene.envs):
        last = []
        for t in (46, 47, 48):
            e.data.qpos[:] = raw['positions'][i, t]; e.data.qvel[:] = raw['velocities'][i, t]
            mujoco.mj_forward(e.model, e.data); last.append(e.facts())
        success.append(all(f['inserted'] for f in last)); checks.append(last)
    return {'success': success, 'successes': sum(success), 'n': len(success),
            'rate': float(np.mean(success)), 'final_three_checks': checks}


def collect(seed, checkpoint, round_, dino):
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    out = R/'training_refresh'/f'seed{seed}'/f'step{round_}'; out.mkdir(parents=True, exist_ok=False)
    net, base = load_adapter(seed, checkpoint=Path(checkpoint)), load_base()
    hashes = {'meta': weight_hash(net), 'base': weight_hash(base)}
    encoder, pca = visual_assets(dino)
    chunks = []; began = time.time()
    for offset in range(0, 384, 32):
        ids = np.arange(offset, offset+32)
        scene = SceneBatch([geometry(2201000+int(i)) for i in ids], 2202000+ids, encoder, pca)
        start = 24100000 + round_*100 + seed*1000
        trace = support(scene, net, base, start+ids)
        ks = torch.tensor([1, 2, 3, 5], device='cuda')[torch.arange(32, device='cuda').add(offset).remainder(4)]
        z = trace['meta_paths'].cuda()[torch.arange(32, device='cuda'), ks]
        raw = query(scene, base, start+500000+ids, net, z, labels=True)
        packet = {'support_visual': trace['visual'], 'support_action': trace['actions'],
                  'query_visual': raw['visual'].float(),
                  'proprio': torch.cat((trace['proprio'], torch.tensor(raw['positions'], dtype=torch.float32)), 1),
                  'query_action': torch.tensor(raw['actions'], dtype=torch.float32),
                  'expert_label': torch.tensor(raw['expert_label'], dtype=torch.float32)}
        chunks.append(packet)
        torch.save({'support': trace, 'query': raw, 'training_domain_ids': ids.tolist(),
                    'query_k': ks.cpu(), 'weights': hashes}, out/f'TRAJECTORY_{offset:03d}.pt')
        scene.close()
        print(json.dumps({'collected': offset+32, 'seed': seed, 'round': round_, 'seconds': time.time()-began}), flush=True)
    torch.save({key: torch.cat([p[key] for p in chunks]) for key in chunks[0]}, out/'BANK.pt')
    assert weight_hash(net) == hashes['meta'] and weight_hash(base) == hashes['base']
    save_json(out/'COMPLETE.json', {'seed': seed, 'round': round_, 'conditions': 384,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(Path(checkpoint)),
        'bank_sha256': digest(out/'BANK.pt'), 'weights_unchanged': True,
        'source': 'Live meta-adapter support actions/updates and live query control; geometry/dynamics from old training pool; new support/query resets.',
        'labels': 'Expert actions after learned action computation, training supervision only.',
        'query_k': 'Balanced 1/2/3/5 prefixes; all five support transitions cached, only selected prefix latent consumed.',
        'runtime_inputs': ['RGB-derived frozen LiLa/PCA', 'controlled peg XYZ', 'previous issued action'],
        'seconds': time.time()-began})
    print('COLLECT_COMPLETE', seed, round_, flush=True)


def evaluate(stratum, seed, dino):
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    assert (R/'FINAL_FREEZE.json').exists()
    spec = json.loads((R/'PROTOCOL.json').read_text())['final']
    ss = spec['seed']+1000*('id', 'position', 'scale').index(stratum)
    n = spec['domains_per_stratum']; out = R/'final'/stratum/f'seed{seed}'
    out.mkdir(parents=True, exist_ok=False); began = time.time()
    base = load_base(); hashes = {'base': weight_hash(base)}
    encoder, pca = visual_assets(dino)
    scene = SceneBatch([geometry(ss+i, stratum) for i in range(n)], np.arange(n)+ss+100000, encoder, pca)
    nets, latents = {}, {}
    if seed:
        nets = {name: load_adapter(seed, name) for name in ('meta', 'history')}
        hashes.update({k: weight_hash(v) for k, v in nets.items()})
        trace = support(scene, nets['meta'], base, np.arange(n)+ss+200000)
        with torch.no_grad():
            hp = nets['history'].history_path(trace['visual'].cuda(), trace['proprio'].cuda(), trace['actions'].cuda())
        for k in KS:
            latents[f'meta_k{k}'] = trace['meta_paths'][:, k].cuda()
            latents[f'history_k{k}'] = hp[k]
        latents['swapped_k3'] = latents['meta_k3'].roll(1, 0)
        trace['latents'] = {k: v.cpu() for k, v in latents.items()}
        trace.update(weights=hashes, seed_base=ss)
        torch.save(trace, out/'SUPPORT_SEALED.pt')
        save_json(out/'SUPPORT_SEAL.json', {'sha256': digest(out/'SUPPORT_SEALED.pt'),
            'query_not_started': True, 'time': time.time()})
        arms = [f'{name}_k{k}' for name in ('meta', 'history') for k in KS] + ['swapped_k3']
    else:
        arms = ['base', 'oracle']
    metrics = {}
    for arm in arms:
        if arm in ('base', 'oracle'): net, z = None, None
        else:
            net = nets['history' if arm.startswith('history') else 'meta']; z = latents[arm]
        raw = query(scene, base, np.arange(n)+ss+300000, net, z, oracle=arm == 'oracle')
        assert raw['expert_label'] is None
        raw.update(seed_base=ss, stratum=stratum, arm=arm)
        torch.save(raw, out/f'{arm}_RAW.pt'); sha = digest(out/f'{arm}_RAW.pt')
        metrics[arm] = score(scene, raw)
        metrics[arm]['raw_sha256'] = sha
        print(json.dumps({'stratum': stratum, 'seed': seed, 'arm': arm,
              'successes': metrics[arm]['successes'], 'n': n, 'seconds': time.time()-began}), flush=True)
    assert weight_hash(base) == hashes['base']
    assert all(weight_hash(net) == hashes[name] for name, net in nets.items())
    save_json(out/'RESULT.json', {'metrics': metrics, 'model_hashes': hashes, 'weights_unchanged': True,
        'slot_geometry_runtime_access': False, 'query_z_fixed': True, 'seconds': time.time()-began,
        'support_policy': 'Same meta-generated live support prefix supplied to all arms within each model seed; seeds have separate learned policies.',
        'frozen_z_alias': 'meta_k0 is the same checkpoint with zero latent updates; support cannot affect it.'})
    scene.close(); print('EVAL_COMPLETE', stratum, seed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['collect', 'evaluate'])
    p.add_argument('--seed', type=int, required=True); p.add_argument('--round', type=int)
    p.add_argument('--checkpoint'); p.add_argument('--stratum', choices=['id', 'position', 'scale'])
    p.add_argument('--dino', default=DEFAULT_MODEL); a = p.parse_args()
    if a.mode == 'collect': collect(a.seed, a.checkpoint, a.round, a.dino)
    else: evaluate(a.stratum, a.seed, a.dino)
