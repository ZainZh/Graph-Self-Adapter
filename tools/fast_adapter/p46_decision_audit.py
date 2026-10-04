"""Bounded P46 decision replay from sealed public encodings; no physics or RGB.

All 1,848 prescribed cells must exist and pass seal/source/checkpoint checks
before any numerical trajectory is replayed. Only the first OOD condition in
each task is replayed: 36 main plus 15 supplemental cells, chosen without
outcomes. This is not an audit of all decisions or of physical execution.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from prove.fast_adapter.geometry3d_adapter import pose
from prove.fast_adapter.p45_neural import P45Latent, load_model
from tools.fast_adapter import p45_evaluate as runtime
from tools.fast_adapter import p46_aggregate as scope
from tools.fast_adapter.p45_common import sha

ROOT = scope.ROOT
TOLERANCES = dict(latent=1e-5, position=1e-6, rotation_matrix=1e-6,
                  geometry=1e-6, controller_state=1e-6, loss=1e-6, diagnostic=1e-6)
REQUIRED = ('RESULT.json', 'PRE_RUN.json', 'SEAL.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return dict(path=str(path), sha256=sha(path))


def selected(cell):
    return cell['split'] == 'geometry' and cell['condition_seed'] == scope.FINAL_BASE[cell['task']] + 100


def complete_inventory(root):
    """Existence-only first pass: never open results, traces, observations or seals."""
    cohorts = []
    for task in scope.TASKS:
        for role, count in (('main', 576), ('supplemental', 40)):
            path = Path(root) / role / task / 'EVALUATION_FREEZE.json'
            if not path.is_file():
                raise RuntimeError('All 1,848 main and supplemental cells must finish before audit')
            frozen = read(path)
            scope._validate_scope(frozen)
            expected = (scope.expected_main_identities(task, 'id') |
                        scope.expected_main_identities(task, 'geometry') if role == 'main'
                        else scope.expected_supplemental_identities(task))
            if (frozen.get('phase') != 'independent_validation' or frozen['cohort_role'] != role
                    or set(frozen['tasks']) != {task} or len(frozen['cells']) != count
                    or {scope.p45._identity(cell) for cell in frozen['cells']} != expected):
                raise ValueError('Audit requires the exact P46 independent task/role/new-seed matrix')
            for index in range(count):
                directory = path.parent / 'cells' / f'cell_{index:06d}'
                if ((directory / 'ABORT.json').exists()
                        or any(not (directory / name).is_file() for name in REQUIRED)):
                    raise RuntimeError('All 1,848 main and supplemental cells must finish before audit')
            cohorts.append(dict(path=path, frozen=frozen))
    return cohorts


class Models:
    """Small frozen heads only; never instantiate or call the visual encoder."""
    def __init__(self):
        self.cache = {}

    def get(self, frozen, cell):
        if cell['arm'] == 'multiview':
            return None
        candidate = runtime.model_binding(frozen, cell)
        checkpoint = candidate['checkpoint']
        if checkpoint['sha256'] not in self.cache:
            runtime.check_binding(checkpoint, frozen['files'])
            model, package = load_model(checkpoint['path'], 'cpu')
            self.cache[checkpoint['sha256']] = model, package
        model, package = self.cache[checkpoint['sha256']]
        if (model.task != cell['task'] or model.representation != cell['rep']
                or package['seed'] != cell['model_seed'] or package['updates'] != 2000
                or model.width != 96 or model.latent_dim != 8):
            raise ValueError('Audit checkpoint identity or training budget mismatch')
        if cell['rep'] != 'geometry' and package['pca'][cell['rep']]['sha256'] != candidate['pca']['sha256']:
            raise ValueError('Audit projection differs from the trained checkpoint')
        return model


def verify_inventory(cohorts, models):
    """All seals verified before the selected 51 TRACE/OBS arrays are loaded."""
    sample, provenance, snapshots, contracts = [], [], set(), {}
    for cohort in cohorts:
        path, frozen = cohort['path'], cohort['frozen']
        if runtime.validate_freeze(path) != frozen:
            raise ValueError('Freeze changed after completion check')
        task, = frozen['tasks']
        contract = dict(environment=frozen['environments'][task], sources=frozen['sources'],
                        models=frozen['tasks'][task], steps=frozen['steps_per_episode'])
        if task in contracts and contract != contracts[task]:
            raise ValueError('Main and supplemental model/source/controller budgets differ')
        contracts[task] = contract
        digest = sha(path)
        provenance.append(dict(freeze=binding(path), n_sealed_cells=len(frozen['cells'])))
        for index, cell in enumerate(frozen['cells']):
            directory = path.parent / 'cells' / f'cell_{index:06d}'
            pre, seal = read(directory / 'PRE_RUN.json'), read(directory / 'SEAL.json')
            candidate = runtime.model_binding(frozen, cell)
            if (pre['cell'] != cell or pre['cell_index'] != index or pre['freeze']['sha256'] != digest
                    or pre['phase'] != 'independent_validation' or pre['steps'] != frozen['steps_per_episode']
                    or pre['environment'] != frozen['environments'][task] or pre['candidate'] != candidate
                    or pre['engineering_gate'] != frozen['engineering_gates'][task]):
                raise ValueError(f'Invalid pre-run identity: {directory}')
            if (seal.get('schema') != 'prove.p45-trajectory-seal.v1'
                    or not {'TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json'} <= set(seal['files'])):
                raise ValueError(f'Incomplete trajectory seal: {directory}')
            for name, expected in seal['files'].items():
                if Path(name).name != name or sha(directory / name) != expected:
                    raise ValueError(f'Changed sealed file: {directory}/{name}')
            model = models.get(frozen, cell)
            expected = runtime.parameters_sha(model)
            if seal['neural_parameters_before'] != expected or seal['neural_parameters_after'] != expected:
                raise ValueError(f'Runtime/checkpoint parameter mismatch: {directory}')
            snapshot = pre['source_snapshot']
            key = snapshot, tuple(sorted(frozen['sources'].items()))
            if key not in snapshots:
                for source, expected in frozen['sources'].items():
                    relative = Path(source)
                    if relative.is_absolute() or '..' in relative.parts or sha(Path(snapshot) / relative) != expected:
                        raise ValueError(f'Changed frozen source snapshot: {snapshot}/{source}')
                snapshots.add(key)
            if selected(cell):
                sample.append(dict(cell=cell, cell_index=index, directory=directory,
                                   frozen=frozen, freeze=path, parameter_sha256=runtime.parameters_sha(model)))
    if len(sample) != 51 or sum(item['frozen']['cohort_role'] == 'main' for item in sample) != 36:
        raise ValueError('Audit sample must be exactly 36 main and 15 supplemental trajectories')
    return sample, provenance


class Comparison:
    def __init__(self):
        self.max_errors = {key: 0. for key in TOLERANCES}
        self.failures = []

    def fail(self, t, field, reason, **details):
        self.failures.append(dict(t=int(t), field=field, reason=reason, **details))

    def compare(self, actual, saved, t, field, category='controller_state'):
        if isinstance(actual, dict) and isinstance(saved, dict):
            if set(actual) != set(saved):
                self.fail(t, field, 'Dictionary keys differ')
            for key in sorted(set(actual) & set(saved)):
                self.compare(actual[key], saved[key], t, f'{field}.{key}', category)
        elif actual is None or saved is None or isinstance(actual, (str, bool, int, np.bool_, np.integer)):
            same_kind = (actual is None and saved is None
                         or isinstance(actual, str) and isinstance(saved, str)
                         or isinstance(actual, (bool, np.bool_)) and isinstance(saved, (bool, np.bool_))
                         or isinstance(actual, (int, np.integer)) and not isinstance(actual, (bool, np.bool_))
                         and isinstance(saved, (int, np.integer)) and not isinstance(saved, (bool, np.bool_)))
            if not same_kind:
                self.fail(t, field, 'Categorical type differs')
            elif actual != saved:
                self.fail(t, field, 'Categorical value differs', actual=str(actual), saved=str(saved))
        elif isinstance(actual, (list, tuple)) and isinstance(saved, (list, tuple)):
            if len(actual) != len(saved):
                self.fail(t, field, 'Sequence length differs')
            for i, (left, right) in enumerate(zip(actual, saved)):
                self.compare(left, right, t, f'{field}[{i}]', category)
        else:
            try:
                left, right = np.asarray(actual, dtype=float), np.asarray(saved, dtype=float)
                if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                    raise ValueError('Shape mismatch or nonfinite numerical field')
                error = float(np.max(np.abs(left-right), initial=0.))
                self.max_errors[category] = max(self.max_errors[category], error)
                if error > TOLERANCES[category]:
                    self.fail(t, field, 'Absolute tolerance exceeded', error=error, tolerance=TOLERANCES[category])
            except (ValueError, TypeError) as exc:
                self.fail(t, field, str(exc))


def replay(trace, observations, adapter, controller_class, budget):
    """One uninterrupted replay; saved later controller states are comparands only."""
    check = Comparison()
    if len(trace) != budget or len(observations) != budget or not len(trace):
        raise ValueError('Sealed trace/observation length differs from the common action budget')
    controller = controller_class.__new__(controller_class)
    controller.load_state_dict(trace[0]['controller_before'])
    for t, (row, obs) in enumerate(zip(trace, observations)):
        check.compare(row['t'], t, t, 'trace.t')
        check.compare(obs['t'], t, t, 'observation.t')
        check.compare(controller.state_dict(), row['controller_before'], t, 'controller_before')
        check.compare(controller.adaptation_frozen, obs['adaptation_frozen'], t, 'adaptation_frozen')
        if obs['corners'] is not None and not controller.adaptation_frozen:
            adapter.update(obs, t)
        estimate = adapter.estimate()
        if estimate is None or row['estimate'] is None:
            check.compare(estimate, row['estimate'], t, 'estimate')
        else:
            for i, category in enumerate(('position', 'rotation_matrix', 'geometry')):
                check.compare(estimate[i], row['estimate'][i], t, f'estimate[{i}]', category)
        update = copy.deepcopy(adapter.records[-1]) if adapter.records and adapter.records[-1]['t'] == t else None
        counterfactual = None
        if update is not None:
            prior = pose(np.asarray(update['pre_update_points']))
            counterfactual = runtime.counterfactual_command(controller, t, prior, row['proprio'])
        p, rotation, closed, stage = controller.command(t, estimate, row['proprio'])
        check.compare(p, row['command_p'], t, 'command_p', 'position')
        check.compare(rotation, row['command_R'], t, 'command_R', 'rotation_matrix')
        check.compare(bool(closed), row['closed'], t, 'closed')
        check.compare(stage, row['stage'], t, 'stage')
        check.compare(controller.state_dict(), row['controller_after'], t, 'controller_after')
        if update is not None:
            p0, r0, closed0, stage0 = counterfactual
            update.update(same_state_action_position_delta=float(np.linalg.norm(p-p0)),
                          same_state_action_rotation_delta=float(np.linalg.norm(
                              Rotation.from_matrix(rotation @ r0.T).as_rotvec())),
                          same_state_closed_changed=bool(closed != closed0),
                          same_state_stage_changed=bool(stage != stage0))
        if update is None or row['adaptation'] is None:
            check.compare(update, row['adaptation'], t, 'adaptation')
        else:
            saved = row['adaptation']
            if set(update) != set(saved):
                check.fail(t, 'adaptation', 'Update record fields differ')
            for key in sorted(set(update) & set(saved)):
                category = ('latent' if key in ('z', 'initial_z', 'delta_z') else
                            'loss' if key in ('support_loss_before', 'support_loss_after', 'inner') else
                            'geometry' if 'points' in key else 'diagnostic')
                check.compare(update[key], saved[key], t, f'adaptation.{key}', category)
    return dict(passed=not check.failures, steps_replayed=budget,
                max_absolute_errors=check.max_errors, failures=check.failures,
                adaptation_events=adapter.update_events,
                geometric_fit_events=getattr(adapter, 'fit_events', 0))


def audit(root, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Preserve existing audit output: {out}')
    cohorts = complete_inventory(root)
    models = Models()
    sample, provenance = verify_inventory(cohorts, models)
    records = []
    for item in sample:
        cell, frozen, directory = item['cell'], item['frozen'], item['directory']
        result = dict(cell=cell, cell_index=item['cell_index'], freeze=binding(item['freeze']),
                      seal=binding(directory / 'SEAL.json'), path=str(directory),
                      checkpoint_parameter_sha256=item['parameter_sha256'],
                      replay_parameters_before=None)
        model = None
        try:
            model = models.get(frozen, cell)
            result['replay_parameters_before'] = runtime.parameters_sha(model)
            candidate = runtime.model_binding(frozen, cell)
            donor = runtime.donor_deltas(cell, candidate) if cell['arm'] == 'mismatched' else None
            adapter = (runtime.MultiviewReference(cell['task']) if model is None
                       else P45Latent(model, cell['arm'], donor_deltas=donor))
            module = importlib.import_module(frozen['environments'][cell['task']]['controller_module'])
            result.update(replay(np.load(directory / 'TRACE.npy', allow_pickle=True),
                                 np.load(directory / 'OBS.npy', allow_pickle=True), adapter,
                                 module.GeometryTaskController, frozen['steps_per_episode']))
        except Exception as exc:
            result.update(passed=False, failures=[dict(reason=str(exc), type=type(exc).__name__)])
        after = runtime.parameters_sha(model)
        result['replay_parameters_after'] = after
        result['replay_parameters_unchanged'] = (result['replay_parameters_before'] == after
                                               == item['parameter_sha256'])
        if not result['replay_parameters_unchanged']:
            result['passed'] = False
            result['failures'].append(dict(reason='Replay changed checkpoint parameters'))
        records.append(result)
    report = dict(schema='prove.p46-decision-audit.v1', passed=all(row['passed'] for row in records),
        audit_source=binding(__file__), campaign=scope.CAMPAIGN, provenance=provenance,
        prerequisites=dict(all_1848_complete_and_sealed=True, source_and_checkpoint_bindings_verified=True),
        sampling='First predeclared OOD condition per task: 12 main plus 5 supplemental; no outcome selection',
        n_total_campaign_cells=1848, n_decision_trajectories_audited=len(records),
        n_main_audited=36, n_supplemental_audited=15, tolerances=TOLERANCES,
        max_absolute_errors={key: max(row.get('max_absolute_errors', {}).get(key, 0.) for row in records)
                             for key in TOLERANCES},
        limitation='51 numerical decision trajectories only; no RGB encoder recomputation, new episodes, '
                   'physics replay, success re-evaluation, or all-1848 decision audit', records=records)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'AUDIT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(args.root, args.out)
    print(json.dumps(dict(passed=report['passed'], sampled=51, out=args.out)))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
