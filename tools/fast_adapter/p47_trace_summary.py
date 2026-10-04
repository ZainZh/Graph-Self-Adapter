"""Read-only descriptive diagnostics, gated on a complete P47 main summary.

CLI: python -m tools.fast_adapter.p47_trace_summary --summary RESULTS.json --out NEW_DIR

Writes TRACE_SUMMARY.json (prove.p47-trace-summary.v1) and MANIFEST.json
(prove.p47-trace-summary-manifest.v1). Groups retain each task/representation/
arm/model, pool the two fixed models, and include task and campaign totals.
Distributions give finite n/median/p95/max and explicit nonfinite/null counts;
Boolean summaries give n/true/false/all_true (null for an empty population).
No simulator, model, evaluator, hypothesis test, or RGB encoder is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = 'p47_representation_solver'
TASK_BASES = {'cup': 480100, 'insert': 481100, 'slider': 482100}
IDENTITY = ('task', 'split', 'condition_seed', 'rep', 'model_seed', 'arm')
OUTCOME_BOOLEANS = ('success', 'validated_success', 'mechanical_valid',
                    'sustained_native_success', 'trajectory_finite',
                    'task_success_denominator_included', 'adaptation_frozen')
COUNT_FIELDS = ('adaptation_events', 'adaptation_updates', 'gradient_steps',
                'direct_solves', 'geometric_fit_events')
SOLVER_NUMBERS = ('rank', 'latent_dim', 'rank_rtol', 'rank_threshold',
                  'condition_number', 'retained_condition_number',
                  'loss_before', 'loss_after_float64', 'loss_after_runtime_z',
                  'gradient_norm', 'delta_z_norm', 'stored_delta_z_norm', 'z_norm',
                  'nullspace_gradient_norm', 'stationarity_residual',
                  'stationarity_tolerance', 'runtime_rounding_norm', 'solve_seconds')
SOLVER_BOOLEANS = ('rank_deficient', 'stationarity_within_tolerance',
                   'recorded_numbers_finite', 'rank_matches_recorded_eigenvalues',
                   'stationarity_flag_matches_recorded_values',
                   'no_material_negative_eigenvalue',
                   'float64_loss_finite_and_nonincreasing',
                   'runtime_z_loss_finite_and_nonincreasing')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(filename):
    digest = hashlib.sha256()
    with path(filename).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(filename):
    return dict(path=str(path(filename).resolve()), sha256=sha(filename))


def verified_bytes(item, expected_path=None):
    require(isinstance(item, dict) and {'path', 'sha256'} <= item.keys(),
            'Missing explicit artifact binding')
    filename = path(item['path'])
    if expected_path is not None:
        require(filename.resolve() == path(expected_path).resolve(), 'Bound artifact path mismatch')
    data = filename.read_bytes()
    require(sha_bytes(data) == item['sha256'], f'Bound artifact changed: {filename}')
    return data


def identity(cell):
    return tuple(cell[key] for key in IDENTITY)


def expected_identities():
    result = set()
    for task, base in TASK_BASES.items():
        for seed in range(base, base + 32):
            result.update((task, 'geometry', seed, rep, model, arm)
                          for rep in ('geometry', 'relative_fusion')
                          for model in (11, 22) for arm in ('fixed', 'continuous', 'direct'))
            result.add((task, 'geometry', seed, 'geometry', None, 'multiview'))
    return result


def validate_summary(summary):
    """Reject incomplete/unknown scope before opening any episode outcome."""
    require(summary.get('schema') == 'prove.p47-summary.v1'
            and summary.get('campaign') == CAMPAIGN
            and summary.get('phase') == 'independent_validation'
            and summary.get('status') == 'complete'
            and summary.get('formal_inference_ready') is True
            and summary.get('prespecified_tasks') == list(TASK_BASES),
            'Requires a complete verified P47 independent-validation summary')
    for key in ('n_prespecified_physical_cells', 'n_planned_cells',
                'n_declared_cells', 'n_completed_cells'):
        require(type(summary.get(key)) is int and summary[key] == 1248,
                f'Complete 1248-cell summary required: {key}')
    for key in ('n_unknown_cells', 'n_undeclared_required_cells', 'n_unexpected_cells'):
        require(type(summary.get(key)) is int and summary[key] == 0, f'Unknown scope: {key}')
    require(summary.get('formal_scope_gaps') == []
            and summary.get('cell_status_counts') == {'completed': 1248},
            'Unknown or incomplete cell evidence')
    evidence = summary.get('evidence')
    require(isinstance(evidence, list) and len(evidence) == 1248, 'All 1248 cell records are required')
    require(all(item.get('status') == 'completed' and {'cell', 'cell_index', 'path', 'binding', 'seal'} <= item.keys()
                for item in evidence), 'Incomplete cell records')
    require({identity(item['cell']) for item in evidence} == expected_identities(),
            'Summary differs from the exact 1248-cell main matrix')
    provenance = summary.get('provenance', [])
    require(len(provenance) == 3 and {item['task'] for item in provenance} == set(TASK_BASES),
            'One verified main freeze per task is required')


def boolean(value):
    require(type(value) in (bool, np.bool_), 'Missing or invalid Boolean diagnostic')
    return bool(value)


def number(value, nullable=False):
    if value is None and nullable:
        return None
    require(isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)),
            'Missing or invalid numeric diagnostic')
    return float(value)


def count(value):
    require(type(value) in (int, np.int64, np.int32) and value >= 0, 'Invalid event/step count')
    return int(value)


def distribution(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return dict(n=len(finite), n_recorded=len(values),
                n_nonfinite=sum(value is not None and not math.isfinite(value) for value in values),
                n_not_applicable=sum(value is None for value in values),
                median=float(np.median(finite)) if finite else None,
                p95=float(np.quantile(finite, .95)) if finite else None,
                max=max(finite) if finite else None)


def booleans(values):
    return dict(n=len(values), true=sum(values), false=len(values) - sum(values),
                all_true=all(values) if values else None)


def finite_nonincrease(before, after):
    # Exact recorded comparison, without a new post-hoc numerical tolerance.
    return math.isfinite(before) and math.isfinite(after) and after <= before


def solver_values(solver):
    require(isinstance(solver, dict), 'Missing direct-solver diagnostics')
    values = {key: number(solver[key], nullable=key in ('condition_number', 'retained_condition_number'))
              for key in SOLVER_NUMBERS}
    rank, dim = count(solver['rank']), count(solver['latent_dim'])
    require(dim == 8 and rank <= dim and values['rank_rtol'] == 1e-6, 'Unknown direct solver contract')
    require((values['condition_number'] is None) == (rank < dim)
            and (values['retained_condition_number'] is None) == (rank == 0),
            'Missing or inapplicable direct condition number')
    eigenvalues = [number(value) for value in solver['hessian_eigenvalues']]
    require(len(eigenvalues) == dim, 'Incomplete direct eigenvalue diagnostics')
    flags = {key: boolean(solver[key]) for key in ('rank_deficient', 'stationarity_within_tolerance')}
    require(flags['rank_deficient'] == (rank < dim), 'Direct rank flag disagrees with recorded rank')
    finite = all(math.isfinite(value) for value in [*eigenvalues, *values.values()] if value is not None)
    flags.update(recorded_numbers_finite=finite,
                 rank_matches_recorded_eigenvalues=rank == sum(value > values['rank_threshold'] for value in eigenvalues),
                 stationarity_flag_matches_recorded_values=flags['stationarity_within_tolerance'] ==
                 (values['stationarity_residual'] <= values['stationarity_tolerance']),
                 no_material_negative_eigenvalue=all(math.isfinite(value) and value >= -values['rank_threshold']
                                                    for value in eigenvalues),
                 float64_loss_finite_and_nonincreasing=finite_nonincrease(values['loss_before'], values['loss_after_float64']),
                 runtime_z_loss_finite_and_nonincreasing=finite_nonincrease(values['loss_before'], values['loss_after_runtime_z']))
    values.update(loss_change_float64=values['loss_after_float64'] - values['loss_before'],
                  loss_change_runtime_z=values['loss_after_runtime_z'] - values['loss_before'],
                  min_hessian_eigenvalue=min(eigenvalues), max_hessian_eigenvalue=max(eigenvalues))
    return values, flags


def episode_statistics(cell, result, trace):
    """Reduce recorded values only; no re-encoding, controller replay or integration."""
    arm = cell['arm']
    require(len(trace) == result['steps'] and [row['t'] for row in trace] == list(range(result['steps'])),
            'Incomplete or noncontiguous trace')
    numbers, flags = defaultdict(list), defaultdict(list)
    counts = Counter(observation_records=0, adaptation_events=0, adaptation_updates=0,
                     gradient_steps=0, direct_solves=0, geometric_fit_events=0,
                     adaptation_events_with_zero_steps=0)
    for key in OUTCOME_BOOLEANS:
        flags['episode.' + key].append(boolean(result[key]))
    for key, value in result.items():
        if type(value) is bool and key not in OUTCOME_BOOLEANS and key != 'neural_weights_unchanged':
            flags['episode.' + key].append(value)
    if arm != 'multiview':
        flags['episode.neural_weights_unchanged'].append(boolean(result['neural_weights_unchanged']))
        require(result['neural_weights_unchanged'] is True, 'Neural weights changed')
    else:
        require(result['neural_weights_unchanged'] is None, 'Multiview has no neural weights')
    require(result['validated_success'] == (result['success'] and result['mechanical_valid'])
            and result['task_success_denominator_included'] is True, 'Inconsistent outcome fields')
    outcome = ('validated_success' if result['validated_success'] else
               'mechanical_anomaly' if not result['mechanical_valid'] else 'ordinary_task_failure')
    require(result['outcome_class'] == outcome, 'Inconsistent outcome class')
    numbers['episode.elapsed_seconds'].append(number(result['elapsed_seconds']))
    previous_events = 0
    previous_fits = 0
    for row in trace:
        update = row['adaptation']
        if update is None:
            continue
        require(update['t'] == row['t'], 'Adaptation/trace index mismatch')
        counts['observation_records'] += 1
        steps = count(update['U'])
        counts['adaptation_updates'] += steps
        if arm == 'multiview':
            require(steps == 0, 'Multiview cannot have latent steps')
            event = boolean(update['fit_updated'])
            flags['observation.fit_updated'].append(event)
            require(count(update['fit_events']) == previous_fits + int(event), 'Multiview fit counter mismatch')
            previous_fits += int(event)
            counts['geometric_fit_events'] += int(event)
        else:
            events = count(update['update_events'])
            require(events - previous_events in (0, 1), 'Invalid cumulative adaptation counter')
            event = events > previous_events
            previous_events = events
            require(steps == len(update['inner']) and (event or steps == 0), 'Latent step/event mismatch')
            require(arm != 'fixed' or (not event and steps == 0), 'Fixed latent performed an update')
            require(arm != 'direct' or steps == int(event), 'Direct solve count differs from events')
            counts['adaptation_events'] += int(event)
            counts['adaptation_events_with_zero_steps'] += int(event and steps == 0)
            counts['gradient_steps'] += steps if arm == 'continuous' else 0
            counts['direct_solves'] += int(event) if arm == 'direct' else 0
            before, after = number(update['support_loss_before']), number(update['support_loss_after'])
            for population in ('neural_observation', 'update') if event else ('neural_observation',):
                numbers[population + '.ray_loss_before'].append(before)
                numbers[population + '.ray_loss_after'].append(after)
                numbers[population + '.ray_loss_change'].append(after - before)
                flags[population + '.ray_loss_finite_and_nonincreasing'].append(finite_nonincrease(before, after))
            if arm == 'direct':
                if event:
                    diagnostics, checks = solver_values(update['solver'])
                    for key, value in diagnostics.items():
                        numbers['direct.' + key].append(value)
                    for key, value in checks.items():
                        flags['direct.' + key].append(value)
                else:
                    require(update['solver'] is None, 'Unexpected direct solver without an event')
        for population in ('observation', 'update') if event else ('observation',):
            for key in ('same_state_action_position_delta', 'same_state_action_rotation_delta', 'latent_change'):
                numbers[population + '.' + key].append(number(update[key]))
            for key in ('same_state_closed_changed', 'same_state_stage_changed'):
                flags[population + '.' + key].append(boolean(update[key]))
    for key in COUNT_FIELDS:
        require(count(result[key]) == counts[key], f'TRACE/RESULT count mismatch: {key}')
        numbers['episode.' + key].append(float(counts[key]))
    return dict(cell=cell, outcome=outcome, numbers=numbers, booleans=flags, counts=counts)


def verify_provenance(summary):
    freezes = {}
    for item in summary['provenance']:
        frozen = json.loads(verified_bytes(item['freeze']))
        task = item['task']
        require(frozen.get('schema') == 'prove.p47-evaluation.v1'
                and frozen.get('status') == 'frozen' and frozen.get('root_authorized') is True
                and frozen.get('phase') == 'independent_validation' and frozen.get('campaign') == CAMPAIGN
                and set(frozen.get('tasks', {})) == {task}, 'Unknown result-free main freeze')
        require(len(frozen['cells']) == 416 and {identity(cell) for cell in frozen['cells']} ==
                {ident for ident in expected_identities() if ident[0] == task}, 'Incomplete frozen task matrix')
        freezes[task] = (item, frozen)
    return freezes


def load_episode(record, provenance, frozen):
    directory, cell = path(record['path']), record['cell']
    index = record['cell_index']
    require(type(index) is int and 0 <= index < len(frozen['cells'])
            and frozen['cells'][index] == cell, 'Cell differs from original freeze')
    require(directory.resolve() == (path(provenance['results_root']) / f'cell_{index:06d}').resolve(),
            'Episode path differs from verified summary provenance')
    require(not (directory / 'ABORT.json').exists(), 'Aborted episode remains unknown')
    seal = json.loads(verified_bytes(record['seal'], directory / 'SEAL.json'))
    result_bytes = verified_bytes(record['binding'], directory / 'RESULT.json')
    require(seal.get('schema') == 'prove.p45-trajectory-seal.v1' and seal.get('campaign') == CAMPAIGN
            and seal.get('freeze') == provenance['freeze'], 'Original seal/freeze mismatch')
    require({'TRACE.npy', 'PRE_RUN.json', 'OBS.npy', 'TERMINAL.npy'} <= seal['files'].keys(),
            'Incomplete original trajectory seal')
    # Recheck the original seals for the files this utility consumes. Other
    # campaign integrity checks belong to the complete verified main summary.
    # Retained TRACE bytes eliminate a hash/load race on this pickle file.
    retained = {}
    for name in ('TRACE.npy', 'PRE_RUN.json'):
        retained[name] = verified_bytes(dict(path=str(directory / name), sha256=seal['files'][name]))
    pre = json.loads(retained['PRE_RUN.json'])
    require(pre['cell'] == cell and pre['cell_index'] == index and pre['freeze'] == provenance['freeze']
            and pre['phase'] == 'independent_validation' and pre['steps'] == frozen['steps_per_episode']
            and pre['environment'] == frozen['environments'][cell['task']]
            and pre['solver_contract'] == frozen['solver_contract'], 'PRE_RUN/freeze mismatch')
    reference = cell['arm'] == 'multiview'
    candidate = (dict(method='analytic_visible_multiview', checkpoint=None, pca=None) if reference else
                 frozen['tasks'][cell['task']]['models'][cell['rep']][str(cell['model_seed'])])
    require(pre['candidate'] == candidate, 'PRE_RUN candidate mismatch')
    before, after = seal['neural_parameters_before'], seal['neural_parameters_after']
    require(before == after and (before is None if reference else isinstance(before, str) and len(before) == 64),
            'Neural parameter seal mismatch')
    result = json.loads(result_bytes)
    require(identity(result) == identity(cell) and result['cell_index'] == index
            and result['phase'] == 'independent_validation' and result['steps'] == pre['steps'],
            'RESULT/freeze mismatch')
    trace = np.load(io.BytesIO(retained['TRACE.npy']), allow_pickle=True).tolist()
    stats = episode_statistics(cell, result, trace)
    source = dict(cell=cell, result=record['binding'], seal=record['seal'],
                  trace=dict(path=str(directory / 'TRACE.npy'), sha256=seal['files']['TRACE.npy']),
                  pre_run=dict(path=str(directory / 'PRE_RUN.json'), sha256=seal['files']['PRE_RUN.json']))
    return stats, source


def summarize_group(episodes, group):
    numbers, flags, counts = defaultdict(list), defaultdict(list), Counter()
    for episode in episodes:
        counts.update(episode['counts'])
        for key, values in episode['numbers'].items():
            numbers[key].extend(values)
        for key, values in episode['booleans'].items():
            flags[key].extend(values)
    # Keep explicitly empty populations instead of making an absent measure a zero.
    for key in SOLVER_NUMBERS:
        numbers['direct.' + key]
    for key in SOLVER_BOOLEANS:
        flags['direct.' + key]
    for population in ('neural_observation', 'update'):
        for key in ('ray_loss_before', 'ray_loss_after', 'ray_loss_change'):
            numbers[population + '.' + key]
        flags[population + '.ray_loss_finite_and_nonincreasing']
    for population in ('observation', 'update'):
        for key in ('same_state_action_position_delta', 'same_state_action_rotation_delta', 'latent_change'):
            numbers[population + '.' + key]
        for key in ('same_state_closed_changed', 'same_state_stage_changed'):
            flags[population + '.' + key]
    outcomes = Counter(episode['outcome'] for episode in episodes)
    return dict(**group, n_episodes=len(episodes), n_unknown=0,
                validated_successes=outcomes['validated_success'],
                mechanical_anomalies=outcomes['mechanical_anomaly'],
                ordinary_task_failures=outcomes['ordinary_task_failure'],
                counts=dict(counts), distributions={key: distribution(values) for key, values in sorted(numbers.items())},
                booleans={key: booleans(values) for key, values in sorted(flags.items())})


def grouped(episodes, fields):
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode['cell'][key] for key in fields)].append(episode)
    return [summarize_group(values, dict(zip(fields, key)))
            for key, values in sorted(groups.items(), key=lambda pair: repr(pair[0]))]


def run(summary_path, out):
    out = path(out)
    if out.exists():
        raise FileExistsError(f'Preserve existing output: {out}')
    summary_bytes = path(summary_path).read_bytes()
    summary = json.loads(summary_bytes)
    validate_summary(summary)
    for record in summary['evidence']:
        require(path(record['path']).resolve() not in out.resolve().parents,
                'Derivative output must be outside original episode directories')
    freezes = verify_provenance(summary)
    episodes, sources = [], []
    for record in summary['evidence']:
        provenance, frozen = freezes[record['cell']['task']]
        stats, source = load_episode(record, provenance, frozen)
        episodes.append(stats)
        sources.append(source)
    report = dict(schema='prove.p47-trace-summary.v1', campaign=CAMPAIGN, status='complete',
                  phase='independent_validation', descriptive_only=True, n_episodes=1248,
                  interpretation=dict(
                      elapsed_seconds='Whole episode wall time, including observation, rendering, control, simulation, sealing and evaluation; not pure inference latency',
                      same_state_commands='Recorded fixed-controller request deltas; not executed physical counterfactuals or causal geometry attribution',
                      ray_loss='Recorded before/after values on each event\'s fixed support and current neural offset; not a geometry-accuracy metric',
                      populations='observation includes all recorded adapter observations; update includes neural adaptation events or multiview fits; neural_observation excludes multiview',
                      nonincrease='Exact comparison of recorded values, no new numerical tolerance or hypothesis test',
                      distributions='n counts finite values; nonfinite and inapplicable null values have separate counts; p95 uses linear interpolation',
                      booleans='Counts and all_true describe their recorded population; empty all_true is null',
                      pooled='Descriptive episode/update distributions; models and repeated updates are not independent conditions',
                      operations='Reads sealed records only; no RGB re-encoding, controller replay, physical reintegration or new success evaluation'),
                  groups=grouped(episodes, ('task', 'rep', 'arm', 'model_seed')),
                  pooled_models=grouped(episodes, ('task', 'rep', 'arm')),
                  tasks=grouped(episodes, ('task',)), overall=summarize_group(episodes, {}))
    report_bytes = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    manifest = dict(schema='prove.p47-trace-summary-manifest.v1', campaign=CAMPAIGN, status='complete',
                    summary=dict(path=str(path(summary_path).resolve()), sha256=sha_bytes(summary_bytes)),
                    implementation=binding(__file__), numpy_version=np.__version__,
                    provenance=summary['provenance'], inputs=sources,
                    verification_scope='Requires complete verified main summary; rechecks frozen cell scope, RESULT/SEAL summary bindings and original TRACE/PRE_RUN seals; does not repeat model, source, asset or other artifact audits',
                    result_binding='RESULT.json is bound by the complete summary; the original trajectory seal predates RESULT and does not include it',
                    output_sha256={'TRACE_SUMMARY.json': sha_bytes(report_bytes)})
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'TRACE_SUMMARY.json').open('xb') as handle:
        handle.write(report_bytes)
    with (out / 'MANIFEST.json').open('xb') as handle:
        handle.write(manifest_bytes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, help='Complete verified P47 RESULTS.json')
    parser.add_argument('--out', required=True, help='New derivative output directory')
    args = parser.parse_args()
    report = run(args.summary, args.out)
    print(json.dumps(dict(status=report['status'], n_episodes=report['n_episodes'], out=str(path(args.out)))))


if __name__ == '__main__':
    main()
