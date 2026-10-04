"""Read-only P28 recomputation from sealed float32 runtime observations."""
import json
import numpy as np
import torch
from tools.fast_adapter.p28_common import R, TASKS, SEEDS, task_root, digest, save_json, load_base, load_adapter, TaskEnv, transform
from tools.fast_adapter.p28_eval import EVAL_STRATA
from tools.fast_adapter.p28_train import support_inputs


def maxerror(a, b):
    return float((a-b).abs().max())


def recorded_public(row):
    return torch.cat((torch.zeros_like(row['actions'][:, :1]), row['actions']), 1)


def main():
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    done = json.loads((R/'CAMPAIGN_DONE.json').read_text())
    errors = dict(base=0., action=0., latent=0., support=0.)
    counts = dict(queries=0, support_trajectories=0, physical_replay_states=0, source_hashes=0)
    pred_records = []
    for freeze in ('PREPARATION_FREEZE.json', 'EVALUATION_SOURCE_FREEZE.json', 'ANALYSIS_FREEZE.json'):
        for b in json.loads((R/freeze).read_text())['bindings']:
            assert digest(b['path']) == b['sha256'], b['path']
            counts['source_hashes'] += 1
    for task in done['qualified_tasks']:
        root = task_root(task); base = load_base(task)
        nets = {(s, k): load_adapter(task, s, k) for s in SEEDS for k in ('fast', 'history')}
        for b in json.loads((root/'MODEL_FREEZE.json').read_text())['bindings']:
            assert digest(b['path']) == b['sha256'], b['path']
            counts['source_hashes'] += 1
        # Static controls are exactly means of TRAINING latent records.
        bank_paths = sorted((root/'data/meta').glob('support_[0-9]*.pt'))
        assert len(bank_paths) == 4
        for seed in SEEDS:
            net, blob = nets[(seed, 'fast')]; train_z = []
            for bp in bank_paths:
                b = torch.load(bp, weights_only=True)
                z, _ = net.adapt(support_inputs({k: b[k].cuda() for k in ('visual', 'proprio', 'actions')}), 3)
                train_z.append(z.cpu())
            error = maxerror(torch.cat(train_z), blob['training_z'])
            errors['latent'] = max(errors['latent'], error)
            errors['latent'] = max(errors['latent'], maxerror(torch.cat(train_z).mean(0), blob['mean_z']))
        for si, st in enumerate(EVAL_STRATA):
            folder = root/'evaluation'/st
            result = json.loads((folder/'RESULT.json').read_text())
            assert result['networks_unchanged'] and result['initial_states_matched']
            origin = 4900000+100000*TASKS.index(task)+20000*si
            for offset in range(0, 64, 16):
                seal = json.loads((folder/f'QUERY_START_SEAL_{offset:03d}.json').read_text())
                for binding in [*seal['support_bindings'], seal['latents']]:
                    assert digest(binding['path']) == binding['sha256']
                    counts['source_hashes'] += 1
                latent = torch.load(seal['latents']['path'], weights_only=True)
                source = {}
                for name, delta in [('correct', 2000000), ('donor', 4000000)]:
                    b = torch.load(folder/f'SUPPORT_{name}_{offset:03d}.pt', weights_only=True)
                    assert b['seeds'] == (origin+offset+np.arange(16)+delta).tolist()
                    p = recorded_public(b)
                    with torch.no_grad():
                        a = base(b['visual'][:, :-1].cuda(), b['proprio'][:, :-1].cuda(), p[:, :-1].cuda())
                    errors['support'] = max(errors['support'], maxerror(a.cpu(), b['actions']))
                    source[name] = support_inputs({k: b[k].cuda() for k in ('visual', 'proprio', 'actions')})
                    counts['support_trajectories'] += 16
                for seed in SEEDS:
                    fast, blob = nets[(seed, 'fast')]
                    for name in ('correct', 'donor'):
                        z, _ = fast.adapt(source[name], 3)
                        errors['latent'] = max(errors['latent'], maxerror(z.cpu(), latent[f'fast_{name}_seed{seed}']))
                    errors['latent'] = max(errors['latent'], maxerror(blob['mean_z'][None].expand(16, -1), latent[f'fast_mean_seed{seed}']))
                    history, _ = nets[(seed, 'history')]
                    z, _ = history.adapt(source['correct'], 3)
                    errors['latent'] = max(errors['latent'], maxerror(z.cpu(), latent[f'history_seed{seed}']))
                b = torch.load(folder/f'base_{offset:03d}.pt', weights_only=True)
                initial = b['states']['qpos'][:, 0]
                # Prediction diagnostic on shared actual base states, no tuning.
                for seed in SEEDS:
                    net, _ = nets[(seed, 'fast')]
                    with torch.no_grad():
                        u = net.encode(b['visual'][:, :-1].cuda(), b['proprio'][:, :-1].cuda(), recorded_public(b)[:, :-1].cuda())
                        target = (b['proprio'][:, 1:, :3]-b['proprio'][:, :-1, :3]).cuda()/.01
                        for source_name in ('correct', 'mean', 'donor'):
                            z = latent[f'fast_{source_name}_seed{seed}'].cuda()[:, None].expand(-1, 160, -1)
                            mse = (net.prediction(u, b['actions'].cuda(), z)-target).square().mean((1, 2)).cpu()
                            pred_records.append({'task': task, 'stratum': st, 'seed': seed, 'offset': offset,
                                                 'source': source_name, 'condition_mse': mse.tolist()})
                for row in [x for x in result['rows'] if x['offset'] == offset]:
                    assert digest(row['file']) == row['sha256'], row['file']
                    b = torch.load(row['file'], weights_only=True)
                    assert 'teacher' not in b and b['updates_in_query'] == 0
                    assert torch.equal(b['states']['qpos'][:, 0], initial)
                    assert b['seeds'] == (origin+offset+np.arange(16)+5000000).tolist()
                    prev = recorded_public(b)[:, :-1].cuda()
                    with torch.no_grad():
                        v, p = b['visual'][:, :-1].cuda(), b['proprio'][:, :-1].cuda()
                        ba = base(v, p, prev)
                        errors['base'] = max(errors['base'], maxerror(ba.cpu(), b['base']))
                        if row['arm'] == 'base':
                            action = ba
                        else:
                            seed = int(row['arm'].rsplit('seed', 1)[1]); name = 'history' if row['arm'].startswith('history') else 'fast'
                            net = nets[(seed, name)][0]
                            z = latent[row['arm']].cuda()[:, None].expand(-1, 160, -1)
                            assert torch.equal(latent[row['arm']], b['latent'])
                            action = net.action(v, p, prev, ba, z)
                            k0 = net.action(v, p, prev, ba, net.z0.expand_as(z))
                            assert torch.equal(k0, ba)
                        errors['action'] = max(errors['action'], maxerror(action.cpu(), b['actions']))
                    sidecar = json.loads(__import__('pathlib').Path(row['file']).with_suffix('.json').read_text())
                    assert sidecar['sha256'] == row['sha256']
                    assert row['success'] == [bool(any(x['success'] for x in xs)) for xs in sidecar['metrics']]
                    assert row['terminal_success'] == [bool(xs[-1]['success']) for xs in sidecar['metrics']]
                    # Replay every physical state through the native evaluator.
                    # Runtime observations/seals stay untouched.
                    for j in range(16):
                        e = TaskEnv(task, np.array(sidecar['matrices'][j]), render=False)
                        e.reset(b['seeds'][j])
                        import mujoco
                        replay_success = []
                        for t in range(1, 161):
                            for key in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat'):
                                getattr(e.env.data, key)[:] = b['states'][key][j, t].numpy()
                            mujoco.mj_forward(e.env.model, e.env.data)
                            obs = e.env._get_obs()
                            # Official task success does not depend on requested-vs-executed action;
                            # pass the actual applied command to preserve evaluator semantics.
                            action = b['actions'][j, t-1].numpy().copy()
                            action[:3] = e.matrix@action[:3]
                            _, info = e.env.evaluate_state(obs, np.clip(action, -1, 1))
                            replay_success.append(bool(info['success']))
                            counts['physical_replay_states'] += 1
                        assert replay_success == [bool(x['success']) for x in sidecar['metrics'][j]], (task, st, row['arm'], j)
                        e.close()
                    counts['queries'] += 16
                print('AUDIT_BLOCK', task, st, offset, errors, flush=True)
    assert counts['queries'] == done['query_episodes']
    assert counts['support_trajectories'] == done['support_episodes']
    assert max(errors.values()) < .001, errors
    save_json(R/'summary/AUDIT.json', {'passed': True, 'counts': counts, 'max_errors': errors,
              'tolerance': .001, 'all_native_success_states_replayed': True,
              'scope': 'Float32 runtime inputs, latent/action recomputation, static context provenance, K0 identity, paired initial states and every native success state; no bitwise DINO re-encoding claim.'})
    save_json(R/'summary/PREDICTION_DIAGNOSTIC.json', {'diagnostic_only_no_selection': True, 'records': pred_records})
    print('AUDIT_PASSED', counts, errors, flush=True)


if __name__ == '__main__':
    main()
