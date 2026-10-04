"""Post-result P25 trajectory/label diagnosis; never an input to a controller.

No simulator steps, training, rendering, or new outcome comparisons are executed.
Original experiment artifacts and source freezes remain unchanged.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from prove.fast_adapter.geometry_insert import GeometryInsert, domain, geometry
from prove.fast_adapter.insert3d import HALF, INNER
from tools.fast_adapter.p18_run import digest

ROOT = Path('evaluations/fast_adapter_p25_feedback_v01')
TRAIN = Path('evaluations/fast_adapter_p24_common_prior_v01/training_refresh')


def trajectory_rows(raw, metrics):
    rows = []
    arm, st = raw['arm'], raw['stratum']
    for i in range(len(raw['positions'])):
        e = GeometryInsert(geometry(raw['seed_base'] + i, st), render=False)
        e.set_domain(*domain(raw['seed_base'] + 100000 + i))
        pos, vel = raw['positions'][i], raw['velocities'][i]
        actions, base = raw['actions'][i], raw['base'][i].numpy()
        facts, teacher, aligned = [], [], []
        for p, v in zip(pos, vel):
            e.data.qpos[:] = p
            e.data.qvel[:] = v
            mujoco.mj_forward(e.model, e.data)
            facts.append(e.facts())
            teacher.append(e.expert_action())
            aligned.append(bool(np.all(np.abs(p[:2] - e.xy) <= .65 * (INNER - HALF[:2]) * e.scale)))
        inserted = np.array([f['inserted'] for f in facts])
        inside = np.array([f['inside_xy'] for f in facts])
        assert bool(inserted[-3:].all()) == metrics[arm]['success'][i]
        teacher = np.array(teacher[:48])
        feasible = np.clip(teacher, np.maximum(-1, base-.35), np.minimum(1, base+.35))
        actual_mse = ((actions-feasible)**2).mean(axis=1)
        base_mse = ((base-feasible)**2).mean(axis=1)
        bottom = pos[:, 2] - HALF[2]*e.scale
        first = np.flatnonzero(inserted)
        row = dict(stratum=st, arm=arm, condition=i, success=bool(inserted[-3:].all()),
            base_success=metrics['base']['success'][i],
            first_insert=int(first[0]) if len(first) else None,
            last_insert=int(first[-1]) if len(first) else None,
            final_three_inside_xy=bool(inside[-3:].all()),
            final_three_bottom_above_upper_limit=(bottom[-3:] > .013*e.scale).tolist(),
            final_three_facts=facts[-3:],
            final_xy_error_mm=(1000*(pos[-1, :2]-e.xy)).tolist(),
            final_bottom_mm=(1000*bottom[-3:]).tolist(),
            success_bottom_range_mm=[6.5*e.scale, 13*e.scale],
            action_feasible_teacher_mse=float(actual_mse.mean()),
            base_on_identical_visited_states_mse=float(base_mse.mean()),
            last12_action_feasible_teacher_mse=float(actual_mse[-12:].mean()),
            last12_base_on_identical_visited_states_mse=float(base_mse[-12:].mean()),
            last12_vertical_residual=(actions[-12:, 2]-base[-12:, 2]).tolist(),
            inserted_action_states=int(inserted[:48].sum()),
            inserted_action_states_expert_requests_raise=int((inserted[:48] & ~np.array(aligned[:48])).sum()))
        if st == 'id' and i == 5 and arm in ('base', 'physical_correct_seed11'):
            row['illustrative_timeline'] = [dict(state_index=t,
                xy_error_mm=(1000*(pos[t, :2]-e.xy)).tolist(), bottom_mm=float(1000*bottom[t]),
                inserted=bool(inserted[t]), expert_narrow_alignment=aligned[t],
                action=actions[t].tolist(), base_at_same_state=base[t].tolist(),
                expert_action=teacher[t].tolist()) for t in (16, 20, 24, 36, 47)]
        rows.append(row)
        e.close()
    return rows


def training_label_check(bindings):
    configs = [geometry(2201000+i) for i in range(384)]
    xy = np.array([c['xy'] for c in configs])[:, None, :]
    scale = np.array([c['scale'] for c in configs])[:, None]
    rows = []
    for seed in (11, 22, 33):
        for step in (10000, 30000):
            path = TRAIN/f'seed{seed}'/f'step{step}'/'BANK.pt'
            bank = torch.load(path, weights_only=True)
            p = bank['proprio'][:, 6:54].numpy()
            labels = bank['expert_label'].numpy()
            bottom = p[:, :, 2] - .030*scale
            inside = (np.abs(p[:, :, :2]-xy) <= .0037*scale[:, :, None]).all(-1)
            goal_geometry = inside & (bottom >= .0065*scale) & (bottom <= .013*scale)
            centered = (np.abs(p[:, :, :2]-xy) <= .002275*scale[:, :, None]).all(-1)
            conflict = goal_geometry & ~centered
            rows.append(dict(seed=seed, step=step, action_states=int(goal_geometry.size),
                goal_geometry_states=int(goal_geometry.sum()),
                goal_geometry_but_raise_label=int(conflict.sum()),
                smallest_upward_label=float(labels[:, :, 2][conflict].min())))
            bindings.append(dict(path=str(path), sha256=digest(path)))
    return dict(scope='Geometry-only training label audit. These cached banks do not contain full contact truth; the counts are NOT certified task successes. Labels are read from the actual training bank.',
        rows=rows, total_action_states=sum(r['action_states'] for r in rows),
        goal_geometry_states=sum(r['goal_geometry_states'] for r in rows),
        goal_geometry_but_raise_label=sum(r['goal_geometry_but_raise_label'] for r in rows))


def main():
    torch.set_num_threads(2)
    rows, bindings = [], []
    for st in ('id', 'position', 'scale'):
        folder = ROOT/'development'/st
        result_path = folder/'RESULT.json'
        metrics = json.loads(result_path.read_text())['metrics']
        bindings.append(dict(path=str(result_path), sha256=digest(result_path)))
        for arm in ('base', 'physical_correct_seed11', 'physical_correct_seed22'):
            path = folder/f'{arm}_RAW.pt'
            raw = torch.load(path, weights_only=False)
            rows.extend(trajectory_rows(raw, metrics))
            bindings.append(dict(path=str(path), sha256=digest(path)))
    aggregate = {}
    for st in ('id', 'position', 'scale'):
        selected = [r for r in rows if r['stratum'] == st]
        harms = [r for r in selected if r['arm'] != 'base' and r['base_success'] and not r['success']]
        aggregate[st] = dict(harms_model_condition_runs=len(harms),
            harms_ever_inserted=sum(r['first_insert'] is not None for r in harms),
            harms_final_three_inside_xy=sum(r['final_three_inside_xy'] for r in harms),
            harms_with_at_least_one_final_depth_too_high=sum(any(r['final_three_bottom_above_upper_limit']) for r in harms),
            harms_action_mse_worse_than_base_at_identical_states=sum(
                r['action_feasible_teacher_mse'] > r['base_on_identical_visited_states_mse'] for r in harms),
            base_inserted_action_states=sum(r['inserted_action_states'] for r in selected if r['arm'] == 'base'),
            base_inserted_action_states_expert_requests_raise=sum(
                r['inserted_action_states_expert_requests_raise'] for r in selected if r['arm'] == 'base'))
    training = training_label_check(bindings)
    for name in ('prove/fast_adapter/geometry_insert.py', 'prove/fast_adapter/insert3d.py',
                 'prove/fast_adapter/feedback_adapter.py', 'tools/fast_adapter/p25_train.py',
                 'tools/fast_adapter/p24_rollout.py', __file__):
        path = Path(name)
        if path.is_absolute():
            path = path.relative_to(Path.cwd())
        bindings.append(dict(path=str(path), sha256=digest(path)))
    report = dict(scope='Posthoc diagnosis of opened P25 development records. No new rollouts, no training, no candidate selection; frozen sources/results are not modified.',
        limitations=[
            'Model seeds share the same 12 conditions per stratum; model-condition counts are not independent environments.',
            'Ground-truth geometry/dynamics/contact are used only for post-seal diagnosis, never runtime inputs.',
            'A teacher label can disagree with goal preservation. Its action MSE is not a valid substitute for task success.',
            'Base actions on adapter-visited states are recorded counterfactual actions, not an executed alternative trajectory.',
            'No intervention has yet measured how much each diagnosed issue causes the success-rate loss.',
            'Training geometry counts omit contact validation and cannot be reported as certified successes.'],
        aggregates=aggregate, trajectory_rows=rows, training_labels=training, source_bindings=bindings)
    out = ROOT/'diagnostics'/'POSTHOC_FAILURE_DIAGNOSIS_20260924.json'
    out.parent.mkdir(exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(out), aggregates=aggregate,
        training_label_geometry_conflicts=training['goal_geometry_but_raise_label']), indent=2))


if __name__ == '__main__':
    main()
