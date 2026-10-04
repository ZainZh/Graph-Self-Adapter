"""Post-hoc same-record diagnostics; not additional physical outcomes."""
import json
import numpy as np
import torch
from tools.fast_adapter.p30v2_common import ROOT, TASKS, base_path, dump, digest
from prove.fast_adapter.continuous_adapter import ObjectSkill, ObjectAdapter, feedback


def main(tasks=TASKS):
    torch.set_num_threads(2)
    records = []
    with torch.no_grad():
        for task in tasks:
            base = ObjectSkill(); base.load_state_dict(torch.load(base_path(task), weights_only=True)['state'])
            base.eval()
            for seed in (11, 22):
                nets = {}
                for kind in ('gradient', 'history'):
                    net = ObjectAdapter(history=kind == 'history')
                    net.load_state_dict(torch.load(ROOT/task/'models'/f'seed{seed}'/f'{kind}.pt', weights_only=True)['state'])
                    nets[kind] = net.eval()
                folder = ROOT/task/'development'/f'seed{seed}'
                for split in ('train', 'geometry_ood'):
                    for arm in ('fixed_mean', 'first3', 'gradient', 'history'):
                        bs = [torch.load(p, weights_only=True) for p in sorted(folder.glob(f'{split}_*_{arm}.pt'))]
                        b = {k: torch.cat([v[k] for v in bs]) for k in ('visual', 'proprio', 'actions', 'latent_before_action')}
                        v, nv = b['visual'][:, :-1], b['visual'][:, 1:]
                        p, np_ = b['proprio'][:, :-1], b['proprio'][:, 1:]
                        prev = torch.cat((torch.zeros_like(b['actions'][:, :1]), b['actions'][:, :-1]), 1)
                        net = nets['history' if arm == 'history' else 'gradient']
                        z = b['latent_before_action']; z0 = net.z0.expand_as(z)
                        ba = base(v, p, prev)
                        if task == TASKS[1]: ba[..., 3] = .9
                        correction = b['actions'] - ba
                        y, w = feedback(v, p, nv, np_); u = net.encode(v, p, prev)
                        errs = {}
                        for label, zz in (('current_z', z), ('z0', z0)):
                            err = ((net.prediction(u, b['actions'], zz)-y).square()*w).sum(-1)/w.sum(-1).clamp_min(1)
                            errs[label] = float(err.mean())
                        dist = (z-z0).norm(dim=-1)
                        records.append({'task': task, 'seed': seed, 'split': split, 'arm': arm,
                            'alpha': float(net.alpha), 'z_distance_first20': float(dist[:, :20].mean()),
                            'z_distance_last20': float(dist[:, -20:].mean()),
                            'action_correction_rms_first20': float(correction[:, :20].square().mean().sqrt()),
                            'action_correction_rms_last20': float(correction[:, -20:].square().mean().sqrt()),
                            'same_saved_transition_prediction_mse': errs})
    dump(ROOT/'summary'/('DIAGNOSTICS.json' if tuple(tasks)==TASKS else 'DIAGNOSTICS_'+('_'.join(tasks))+'.json'), {'stage': 'posthoc_development_diagnostic',
        'not_a_counterfactual_success_rate': True,
        'description': 'Recompute prediction with pre-action z versus z0 on identical recorded transitions. Lower prediction error does not imply higher task success. Action correction is relative to base at each saved adaptive state, not a replayed base trajectory.',
        'script_sha256': digest(__file__), 'records': records})
    for r in records:
        if r['arm'] == 'gradient': print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    main()
