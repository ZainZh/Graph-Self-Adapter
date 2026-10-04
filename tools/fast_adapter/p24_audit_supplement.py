"""Post-result supplement: retain original thresholds and report all violations.

The frozen audit stopped at scale/seed33 meta_k1, action error .00525045395.
This append-only diagnostic completes outcome reconstruction and enumerates
precision exceptions. It does not relax the threshold or accept those cases;
separate reconstruction from full-precision rendered inputs is required.
"""
from pathlib import Path
import json
import numpy as np
import torch
import mujoco
from prove.fast_adapter.geometry_insert import GeometryInsert, geometry, domain
from tools.fast_adapter.p24_common import R, KS, SEEDS, load_base, load_adapter, weight_hash
from tools.fast_adapter.p18_run import digest, save_json


def main():
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    assert (R/'CAMPAIGN_DONE.json').exists()
    out = R/'summary'; out.mkdir(exist_ok=True)
    verified = 0
    for name, key in [('SOURCE_FREEZE.json', 'bindings'), ('FINAL_FREEZE.json', 'bindings'), ('EXPORT_RECEIPT.json', 'files')]:
        for b in json.loads((R/name).read_text())[key]:
            assert digest(Path(b['path'])) == b['sha256'], b['path']; verified += 1
    for seed in SEEDS:
        complete = json.loads((R/'models'/f'seed{seed}/COMPLETE.json').read_text())
        assert complete['meta_updates'] == complete['history_updates'] == 50000
        assert sum(complete['k_counts'].values()) == 50000
        assert complete['base_weights_unchanged'] and complete['no_history_warmup']
        assert len(complete['bank_sources']) == 3
        for b in complete['bank_sources']: assert digest(Path(b['path'])) == b['sha256']
    base = load_base('cpu'); episodes = 0; support_steps = 0
    maximum_action_error = 0.; maximum_latent_error = 0.; maximum_support_action_error = 0.
    starts = {}; diagnostics = []; precision_exceptions = []
    for path in sorted((R/'final').glob('*/seed*/RESULT.json')):
        root = path.parent; seed = int(root.name[4:]); st = root.parent.name
        result = json.loads(path.read_text())
        assert result['weights_unchanged'] and result['query_z_fixed']
        assert not result['slot_geometry_runtime_access']
        assert weight_hash(base) == result['model_hashes']['base']
        nets = {}; latents = {}
        if seed:
            trace = torch.load(root/'SUPPORT_SEALED.pt', weights_only=True)
            assert digest(root/'SUPPORT_SEALED.pt') == json.loads((root/'SUPPORT_SEAL.json').read_text())['sha256']
            nets = {name: load_adapter(seed, name, device='cpu') for name in ('meta', 'history')}
            for name, net in nets.items(): assert weight_hash(net) == result['model_hashes'][name]
            assert not any('history_gru' in name for name in nets['meta'].state_dict())
            v, p, a = [trace[k] for k in ('visual', 'proprio', 'actions')]
            mp, losses = nets['meta'].adapt_path(v, p, a)
            with torch.no_grad(): hp = nets['history'].history_path(v, p, a)
            error = float((torch.stack(mp, 1)-trace['meta_paths']).abs().max())
            maximum_latent_error = max(maximum_latent_error, error)
            assert error < .001, (root, 'meta path', error)
            assert torch.allclose(losses, trace['per_transition_loss'], atol=1e-4, rtol=1e-4)
            assert torch.equal(trace['meta_paths'][:, 0], nets['meta'].initial(len(v)))
            # Replay causal support action k from z_k, before seeing observation k+1.
            for t in range(5):
                previous = a[:, t-1] if t else torch.zeros_like(a[:, 0])
                with torch.no_grad():
                    ba = base(v[:, t], p[:, t])
                    expected = nets['meta'].action(v[:, t], p[:, t], previous, ba, trace['meta_paths'][:, t])
                error = float((expected-a[:, t]).abs().max())
                maximum_support_action_error = max(maximum_support_action_error, error)
                assert error < .001, (root, 'support action', t, error)
                support_steps += len(v)
            for k in KS:
                latents[f'meta_k{k}'] = mp[k]
                latents[f'history_k{k}'] = hp[k]
            latents['swapped_k3'] = mp[3].roll(1, 0)
            for arm, z in latents.items():
                error = float((z-trace['latents'][arm]).abs().max())
                maximum_latent_error = max(maximum_latent_error, error)
                assert error < .001, (root, arm, error)
            # Entire old/new interaction representation remains distinct across observations.
            with torch.no_grad():
                previous = torch.cat((torch.zeros_like(a[:, :1]), a), 1)
                u = nets['meta'].encode(v, p, previous)
            diagnostics.append({'stratum': st, 'seed': seed,
                'mean_representation_std': float(u.flatten(0, 1).std(0).mean()),
                'alpha': float(nets['meta'].alpha),
                'z_rms_k3': float((mp[3]-mp[0]).square().mean().sqrt()),
                'z_rms_k5': float((mp[5]-mp[0]).square().mean().sqrt()),
                'one_transition_loss_before': losses[:, :, 0].mean(0).tolist(),
                'one_transition_loss_after': losses[:, :, 1].mean(0).tolist()})
        for arm, metrics in result['metrics'].items():
            rawpath = root/f'{arm}_RAW.pt'; assert digest(rawpath) == metrics['raw_sha256']
            raw = torch.load(rawpath, weights_only=False)
            assert raw['expert_label'] is None
            assert np.max(np.abs(raw['actions'])) <= 1.000001
            if st not in starts: starts[st] = raw['positions'][:, 0].copy()
            else: assert np.array_equal(starts[st], raw['positions'][:, 0]), (root, arm, 'query reset')
            if arm != 'oracle':
                v = raw['visual'][:, :48].float()
                p = torch.tensor(raw['positions'][:, :48], dtype=torch.float32)
                previous = torch.tensor(np.concatenate((np.zeros_like(raw['actions'][:, :1]), raw['actions'][:, :-1]), 1))
                with torch.no_grad():
                    ba = base(v, p)
                    base_difference = (ba-raw['base']).abs()
                    if float(base_difference.max()) >= .005:
                        precision_exceptions.append({'root': str(root), 'arm': arm, 'kind': 'base',
                            'max_error': float(base_difference.max()),
                            'indices': (base_difference >= .005).nonzero().tolist()})
                    if arm == 'base': predicted = ba
                    else:
                        net = nets['history' if arm.startswith('history') else 'meta']
                        z = trace['latents'][arm][:, None, :].expand(-1, 48, -1)
                        predicted = net.action(v, p, previous, raw['base'], z)
                        assert np.max(np.abs(raw['actions']-raw['base'].numpy())) <= .350001
                error = float(np.max(np.abs(predicted.numpy()-raw['actions'])))
                maximum_action_error = max(maximum_action_error, error)
                if error >= .005:
                    precision_exceptions.append({'root': str(root), 'arm': arm, 'kind': 'action',
                        'max_error': error,
                        'indices': np.argwhere(np.abs(predicted.numpy()-raw['actions']) >= .005).tolist()})
            for i in range(metrics['n']):
                env = GeometryInsert(geometry(raw['seed_base']+i, st), render=False)
                env.set_domain(*domain(raw['seed_base']+100000+i))
                checks = []
                for t in (46, 47, 48):
                    env.data.qpos[:] = raw['positions'][i, t]; env.data.qvel[:] = raw['velocities'][i, t]
                    mujoco.mj_forward(env.model, env.data); checks.append(env.facts()['inserted'])
                assert all(checks) == metrics['success'][i], (root, arm, i)
                env.close(); episodes += 1
    assert episodes == 6720
    report = {'passed': not precision_exceptions, 'all_outcomes_recomputed': True, 'precision_exceptions': precision_exceptions, 'source_export_bindings': verified, 'query_episodes_recomputed': episodes,
        'support_actions_recomputed': support_steps, 'maximum_action_replay_error': maximum_action_error,
        'maximum_latent_replay_error': maximum_latent_error,
        'maximum_support_action_replay_error': maximum_support_action_error,
        'tolerances': {'query_action': .005, 'latent': .001, 'support_action': .001},
        'quantization': 'Query projected visuals cached fp16; support projected visuals float32. Live controller used float32.',
        'common_prior_verified': True, 'causal_support_action_order_verified': True,
        'all_query_starts_paired': True, 'network_weights_unchanged': True,
        'diagnostics': diagnostics}
    save_json(out/'AUDIT_SUPPLEMENT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'diagnostics'}), flush=True)


if __name__ == '__main__':
    main()
