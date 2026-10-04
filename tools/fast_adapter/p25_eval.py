"""Small fresh development set. Shared base exploration, sealed frozen-z queries."""
import argparse
import json
import time
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import geometry
from tools.fast_adapter.p24_rollout import SceneBatch, visual_assets, score
from tools.fast_adapter.p25_common import R, VARIANTS, SEEDS, load_base, load_adapter, weight_hash, select_feedback, transitions
from tools.fast_adapter.p18_run import save_json, digest


def trajectory(scene, base, resets, steps, net=None, z=None, oracle=False, bounded=False):
    scene.reset(resets)
    previous = torch.zeros(len(resets), 3, device='cuda')
    ps, vs, velocities, aa, bb, rgbs = [], [], [], [], [], []
    for t in range(steps+1):
        ps.append(np.stack([e.state() for e in scene.envs]))
        velocities.append(np.stack([e.data.qvel.copy() for e in scene.envs]))
        if not oracle or bounded:
            v, p = scene.observe(); vs.append(v.cpu())  # actual float32 runtime input
            if t in (0, steps//2, steps): rgbs.append(scene.last_rgb[:2].copy())
            with torch.no_grad(): ba = base(v, p)
        if t == steps: break
        if oracle:
            action = np.stack([e.expert_action() for e in scene.envs]).astype(np.float32)
            if bounded:
                b = ba.cpu().numpy(); action = np.clip(action, np.maximum(-1, b-.35), np.minimum(1, b+.35))
        else:
            with torch.no_grad(): action_t = ba if net is None else net.action(v, p, previous, ba, z)
            action = action_t.cpu().numpy(); previous = action_t
        if not oracle or bounded: bb.append(ba.cpu())
        aa.append(action); scene.step(action)
    return {'positions': np.stack(ps, 1), 'velocities': np.stack(velocities, 1),
        'actions': np.stack(aa, 1), 'visual': torch.stack(vs, 1) if vs else None,
        'base': torch.stack(bb, 1) if bb else None, 'expert_label': None,
        'audit_rgb': np.stack(rgbs) if rgbs else None, 'audit_rgb_steps': [0, steps//2, steps]}


def main(stratum, dino):
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    assert (R/'MODEL_FREEZE.json').exists()
    spec = json.loads((R/'PROTOCOL.json').read_text())['development']
    n = spec['conditions_each']; ss = spec['seed']+1000*('id','position','scale').index(stratum)
    out = R/'development'/stratum; out.mkdir(parents=True, exist_ok=False)
    base = load_base(); bh = weight_hash(base)
    encoder, pca = visual_assets(dino)
    scene = SceneBatch([geometry(ss+i, stratum) for i in range(n)], ss+100000+np.arange(n), encoder, pca)
    began = time.time()
    support = trajectory(scene, base, ss+200000+np.arange(n), 32)
    p = torch.tensor(support['positions'], dtype=torch.float32)
    actions = torch.tensor(support['actions'])
    idx = select_feedback(p, actions)
    feedback = transitions(support['visual'], p, actions, idx)
    early = transitions(support['visual'], p, actions, torch.arange(3).expand(n, -1))
    support.update(selected_indices=idx, feedback=feedback, early_feedback=early, seed_base=ss)
    torch.save(support, out/'SUPPORT_SEALED.pt')
    save_json(out/'SUPPORT_SEAL.json', {'sha256': digest(out/'SUPPORT_SEALED.pt'),
        'query_not_started': True, 'support_actions_per_condition': 32,
        'consumed_feedback': 3, 'selector_reads': ['controlled peg XYZ change', 'issued action', 'elapsed window'],
        'selector_has_goal_or_contact_truth': False})
    metrics = {}; weights = {'base': bh}; latent_records = {}
    def run(arm, net=None, z=None, oracle=False, bounded=False):
        raw = trajectory(scene, base, ss+300000+np.arange(n), 48, net, z, oracle, bounded)
        raw.update(seed_base=ss, stratum=stratum, arm=arm)
        path = out/f'{arm}_RAW.pt'; torch.save(raw, path)
        metric = score(scene, raw); metric['raw_sha256'] = digest(path)
        metrics[arm] = metric
        print(json.dumps({'stratum': stratum, 'arm': arm, 'successes': metric['successes'], 'n': n}), flush=True)
    run('base'); run('oracle', oracle=True); run('bounded_oracle', oracle=True, bounded=True)
    ft = {k: v.cuda() for k, v in feedback.items()}
    et = {k: v.cuda() for k, v in early.items()}
    for seed in SEEDS:
        nets = {name: load_adapter(seed, name) for name in VARIANTS}
        all_latents = {}; paths = {}
        for name, net in nets.items():
            weights[f'{name}_seed{seed}'] = weight_hash(net)
            path, loss = net.feedback_path(ft)
            checkpoint = torch.load(R/'models'/f'seed{seed}'/f'{name}.pt', map_location='cpu', weights_only=True)
            all_latents[name] = {'correct': path[-1], 'swapped': path[-1].roll(1, 0),
                'mean': checkpoint['training_mean_z'].cuda().expand(n, -1)}
            paths[name] = {'path': torch.stack(path, 1).detach().cpu(),
                           'loss': loss.cpu() if loss is not None else None}
        paths['physical_early'], _ = nets['physical'].feedback_path(et)
        all_latents['physical']['early'] = paths['physical_early'][-1]
        paths['physical_early'] = torch.stack(paths['physical_early'], 1).detach().cpu()
        all_latents['original']['k0'] = nets['original'].initial(n)
        sealed = {'paths': paths, 'latents': {name: {k: v.detach().cpu() for k,v in modes.items()} for name,modes in all_latents.items()}}
        latent_path = out/f'LATENTS_seed{seed}.pt'; torch.save(sealed, latent_path)
        save_json(out/f'LATENTS_seed{seed}_SEAL.json', {'sha256': digest(latent_path), 'adapter_queries_not_started': True})
        latent_records[str(seed)] = digest(latent_path)
        for name, modes in all_latents.items():
            for mode, z in modes.items():
                run(f'{name}_{mode}_seed{seed}', nets[name], z)
        for name, net in nets.items(): assert weight_hash(net) == weights[f'{name}_seed{seed}']
    assert weight_hash(base) == bh
    save_json(out/'RESULT.json', {'metrics': metrics, 'weights': weights, 'weights_unchanged': True,
        'latents_fixed_during_query': True, 'slot_geometry_not_runtime_input': True,
        'k0_aliases': ['anchored_k0=base', 'physical_k0=base', 'history_k0=base'],
        'latent_records': latent_records, 'seconds': time.time()-began})
    scene.close(); print('DEV_COMPLETE', stratum, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stratum', choices=['id','position','scale'], required=True)
    p.add_argument('--dino', default='assets/dinov3'); a = p.parse_args()
    main(a.stratum, a.dino)
