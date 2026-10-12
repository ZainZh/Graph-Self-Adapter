"""Result-free contract tests; no task scene or rollout is started."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

cohort = module('test_v02_cohort', ROOT / 'tools/task_graph/twin_v02_cohort.py')
runner = module('test_v02_runner', ROOT / 'tools/task_graph/twin_rsi.py')
TASKS = ['put_knife_board_new', 'put_mug_on_box', 'sorting_cube_bimanual']

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)); return path

def binding(path):
    return {'path': str(path), 'sha256': cohort.sha256(path)}

@pytest.fixture
def campaign(tmp_path):
    snapshot = tmp_path / 'source'
    code = ['prove/task_graph/twin_graph.py', 'prove/task_graph/twin_physics.py',
            'prove/fast_adapter/twin_local_skill.py', 'tools/task_graph/twin_rsi.py',
            'tools/task_graph/twin_train_skills.py']
    hashes = {}
    for relative in code:
        dest = snapshot / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest); hashes[relative] = cohort.sha256(dest)
    write(snapshot / 'SOURCE_FREEZE.json', hashes)
    runtime = tmp_path / 'runtime.json'
    write(runtime, runner.load_json(ROOT / 'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json'))
    scope = write(tmp_path / 'scope.json', {})
    selection_rule = write(tmp_path / 'selection_rule.json', {})
    rows, tasks = [], []
    for task in TASKS:
        reconstruction = write(tmp_path / task / 'reconstruction.json', {'objects': [
            {'name': name} for name in ['red_knife', 'blue_knife', 'knife_holder', 'board']]})
        tasks.append({'task': task, 'sources': {'reconstruction.json': binding(reconstruction)}})
        sol = tmp_path / 'packages' / ('twin_' + task) / 'solution'
        graph = write(sol / 'graph/task_graph.json', {})
        model = sol / 'models/local_skill.npz'; model.parent.mkdir(); model.write_bytes(b'frozen')
        refs = write(sol / 'refs/visible_centroids.json', {'frames': {'0': {}}, 'runtime_materialization': {
            'cup_release_control': 'portable_tcp_displacement_v1',
            'runtime_contract_canonical_sha256': runner.canonical_json_sha256(cohort.load(runtime))}})
        freeze = write(sol / 'CANDIDATE_FREEZE.json', {'task': task, 'candidate_id': 'v02_b0',
            'graph_sha256': cohort.sha256(graph), 'model_sha256': cohort.sha256(model),
            'references_sha256': cohort.sha256(refs), 'runtime_contract_sha256': cohort.sha256(runtime), 'code': {k: hashes[k] for k in code[:-1]}})
        write(sol / 'MANIFEST.json', {'task': task, 'candidate_id': 'v02_b0',
            'graph_sha256': cohort.sha256(graph), 'references_sha256': cohort.sha256(refs),
            'model': {'sha256': cohort.sha256(model)}, 'candidate_freeze': {'sha256': cohort.sha256(freeze)}})
        write(sol / 'EXTERNAL_BINDINGS.json', {'sources': {'reconstruction.json': binding(reconstruction)}})
        rows.append({'task': task, 'candidate_id': 'v02_b0', 'solution': str(sol), 'source_snapshot': str(snapshot)})
    plan = write(tmp_path / 'plan.json', {'tasks': tasks})
    assets = write(tmp_path / 'assets.json', {'task_assets': {t: {'external_asset_hashes': {}} for t in TASKS}})
    value = {'schema': cohort.SCHEMA, 'status': 'frozen', 'scope_binding': binding(scope),
        'root_plan': binding(plan), 'external_assets': binding(assets), 'runtime_contract': binding(runtime),
        'selection_rule': binding(selection_rule), 'tasks': TASKS, 'trials': list(range(10)),
        'condition_set': 'development_v02', 'condition_seed': 51005, 'execution_authorized': True,
        'output_root': str(tmp_path / 'twin_rsi_v02' / 'cohort'),
        'arms': [{'arm_id': 'B0', 'controller': 'fixed_z', 'tasks': rows}]}
    path = write(tmp_path / 'campaign.json', value)
    return path, value


def test_portable_cli_and_bound_contract():
    for command in ['prepare', 'repackage']:
        argv = [command, '--candidate-id', 'v02_b0', '--runtime-contract',
                str(ROOT / 'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json')]
        if command == 'repackage':
            argv += ['--source-packages', '/tmp/source', '--overrides', '/tmp/override', '--reference-packages', '/tmp/shared']
        assert runner.parser().parse_args(argv).runtime_contract.name == 'RUNTIME_CONTRACT.json'
    overrides = runner.load_json(ROOT / 'evaluations/twin_rsi_v02/B0_OVERRIDES.json')
    with pytest.raises(ValueError, match='explicitly'):
        runner.load_bound_runtime_contract(None, overrides)


def test_source_projection_preserves_raw_and_lift_height():
    raw = np.array([.005, -.006, .1])
    projected, record = runner.project_source_grounding(raw, 'fixed_support_xy_z0')
    assert projected.tolist() == [.005, -.006, 0.]
    assert record['raw_delta_m'] == raw.tolist()
    reference_lift_height = .2
    assert reference_lift_height + projected[2] == .2
    assert raw[2] == .1


def test_cohort_pins_actual_sources_and_conditions(campaign):
    path, value = campaign; plan = cohort.build_plan(path)
    assert len(plan['jobs']) == 30
    assert all(job['condition_sha256'] for job in plan['jobs'])
    labels = [row['label'] for row in plan['bindings']]
    assert labels.count('source_code') == 5
    assert 'selection_rule' in labels
    knife = plan['jobs'][1]
    assert knife['condition_set'] == 'development_v02'
    condition = runner.condition_manifest(runner.load_json(Path(value['root_plan']['path']))['tasks'][0],
        seed=51005, trial=1, nominal=False, condition_design=cohort.load(Path(value['runtime_contract']['path']))['condition_design'])
    shift = condition['actual_initial_translation_m']
    assert shift['red_knife'] == shift['blue_knife'] == shift['knife_holder']
    assert 'ood' not in condition['family']


def test_source_bytes_mutation_rejected(campaign):
    path, value = campaign
    source = Path(value['arms'][0]['tasks'][0]['source_snapshot']) / 'prove/task_graph/twin_physics.py'
    source.write_text(source.read_text() + '\n# mutated\n')
    with pytest.raises(ValueError, match='snapshot code mismatch'):
        cohort.build_plan(path)


@pytest.mark.parametrize('kind', ['empty_arms', 'duplicate_arms', 'duplicate_tasks', 'fresh_seed'])
def test_invalid_campaign_rejected(campaign, kind):
    path, value = campaign
    if kind == 'empty_arms': value['arms'] = []
    if kind == 'duplicate_arms': value['arms'].append(value['arms'][0])
    if kind == 'duplicate_tasks': value['arms'][0]['tasks'].append(value['arms'][0]['tasks'][0])
    if kind == 'fresh_seed': value['condition_seed'] = 92005
    write(path, value)
    with pytest.raises(ValueError): cohort.build_plan(path)


def test_output_equality_and_worker_failure_preserves_denominator(campaign, monkeypatch, tmp_path):
    path, value = campaign; plan = cohort.build_plan(path)
    with pytest.raises(ValueError, match='must equal'):
        cohort.execute(plan, tmp_path / 'wrong', 1)
    def fail(job): raise OSError('worker launch failed')
    monkeypatch.setattr(cohort, '_run_job', fail)
    output = Path(value['output_root'])
    with pytest.raises(RuntimeError, match='incomplete'):
        cohort.execute(plan, output, 2)
    summary = cohort.load(output / 'EXECUTION_SUMMARY.json')
    assert summary['declared_jobs'] == summary['completed_jobs'] == summary['errors'] == 30
    assert summary['task_denominators'] == dict.fromkeys(TASKS, 10)


def test_json_native_warning_tail_retained():
    warning = 'WARNING: Nan, Inf or huge value in QACC at DOF 26. The simulation is unstable. Time = 0.2290.'
    payload, warnings = cohort.parse_stdout('{"output":"x"}\n' + warning)
    assert payload['output'] == 'x' and warnings == [warning]
    with pytest.raises(ValueError): cohort.parse_stdout('{"output":"x"}\nunknown')


def test_fresh_conditions_require_frozen_selection_first(campaign, monkeypatch, tmp_path):
    path, value = campaign
    value['condition_set'] = 'test_v02'; value['condition_seed'] = 92005
    selection = write(tmp_path / 'selection.json', {'status': 'draft'})
    value['selection_freeze'] = binding(selection); write(path, value)
    loaded = []
    original = cohort.load_module
    def track(path, name):
        loaded.append(name); return original(path, name)
    monkeypatch.setattr(cohort, 'load_module', track)
    with pytest.raises(ValueError, match='before condition generation'):
        cohort.build_plan(path)
    assert not any(name.startswith('_v02_runner_') for name in loaded)


@pytest.mark.parametrize('mode', ['wrong_trial', 'wrong_seed', 'wrong_hash', 'warning_closed', 'warning_unclosed'])
def test_report_condition_integrity_and_warning_failclosed(campaign, monkeypatch, mode):
    path, value = campaign; job = cohort.build_plan(path)['jobs'][1]
    output = Path(job['output']); output.mkdir(parents=True)
    task = runner.load_json(Path(value['root_plan']['path']))['tasks'][0]
    condition = runner.condition_manifest(task, seed=51005, trial=1, nominal=False,
        condition_design=cohort.load(Path(value['runtime_contract']['path']))['condition_design'])
    if mode == 'wrong_trial': condition['trial'] = 2
    if mode == 'wrong_seed': condition['seed'] = 92005
    report = {'task': job['task'], 'candidate_id': job['candidate_id'], 'controller': job['controller'],
        'condition_set': job['condition_set'], 'condition': condition,
        'bindings': {'condition_sha256': job['condition_sha256'], 'runtime_contract_sha256': job['runtime_contract_sha256']},
        'strict_success': mode != 'warning_closed', 'original_task_outcome': {'numerical_safety': {'passed': mode != 'warning_closed'}}}
    if mode == 'wrong_hash': report['bindings']['condition_sha256'] = '0' * 64
    report_path = write(output / 'report.json', report)
    stdout = json.dumps({'output': str(report_path)})
    if mode.startswith('warning_'):
        stdout += '\nWARNING: Nan, Inf or huge value in QACC at DOF 26. The simulation is unstable. Time = 0.2290.\n'
    monkeypatch.setattr(cohort.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=stdout, stderr='', returncode=0))
    result = cohort._run_job(job)
    if mode == 'warning_closed':
        assert result['error'] is None and len(result['native_warning_tail']) == 1
    else:
        assert result['error'] is not None
    assert len(list(output.glob('*.stdout'))) == 1


def test_a2_graph_timing_budget_and_default_exact_compatibility():
    plan = ROOT / 'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json'
    b0 = runner.GRAPH.compile_root_plan(plan, ROOT / 'evaluations/twin_rsi_v02/B0_OVERRIDES.json')
    a2 = runner.GRAPH.compile_root_plan(plan, ROOT / 'evaluations/twin_rsi_v02/A2_OVERRIDES.json')
    for task in TASKS:
        frozen = runner.load_json(ROOT / f'task_packages/twin_rsi_v02/B0/packages/twin_{task}/solution/graph/task_graph.json')
        assert b0[task] == frozen
        for parent, stage in zip(b0[task]['stages'], a2[task]['stages']):
            speed = {'Grasp': .5, 'Lift': .5, 'Align': .7}.get(stage['phase'], 1.)
            assert stage.get('reference_speed_scale', 1.) == speed
            assert stage['source_grounding_plane'] == 'none'
            duration = stage['reference']['end_time_sec'] - stage['reference']['start_time_sec']
            slack = parent['max_budget_commands'] - parent['reference_commands']
            confirmation = 2 if stage['phase'] == 'Grasp' else 0
            assert stage['max_budget_commands'] == max(2, int(np.ceil(duration / speed * 10))) + slack + confirmation
            assert stage['reference'] == parent['reference']
            assert stage['invariant_predicates'] == parent['invariant_predicates']
            if confirmation:
                assert stage['completion_predicates'][:-1] == parent['completion_predicates']
                assert stage['completion_predicates'][-1]['predicate'] == 'holding_confirmed'


@pytest.mark.parametrize('speed', [1., .5, .7])
def test_scaled_reference_clock_chain_rule_and_physical_correction(speed):
    clock = runner.StageReferenceClock(1., 3., speed)
    assert clock.at(.4) == pytest.approx(1. + .4 * speed)
    assert clock.duration(2. / speed - .04, .1) == pytest.approx(.04)
    assert clock.at(10.) == 3. and clock.progress(10.) == 1.
    previous = np.full((2, 6), .12)
    delta = np.full((2, 6), .02)
    def call(elapsed):
        return runner.reference_motion_command(clock, stage_elapsed=.4, command_elapsed=elapsed,
            command_duration=.1, previous_correction=previous, correction_delta=delta,
            joint_curve=lambda t: np.full((2, 6), t*t),
            joint_velocity=lambda t: np.full((2, 6), 2*t), grip_curve=lambda t: np.array([t,t]),
            trajectory_start=0., trajectory_end=4.)
    q, grip, velocity = call(.04)
    query = clock.at(.44)
    assert np.allclose(q, query*query + previous + .4*delta)
    assert np.allclose(grip, query)
    assert np.allclose(velocity, 2*query*speed + delta/.1)
    numerical_velocity = (call(.040001)[0]-call(.039999)[0])/.000002
    assert np.allclose(velocity, numerical_velocity)


def test_grasp_confirmation_false_unknown_and_stage_reset():
    gate = runner.HoldingConfirmation()
    assert gate.update(True, 1., .2) is False
    assert gate.update(True, 1.1, .2) is False
    assert gate.update(False, 1.15, .2) is False and gate.elapsed == 0.
    assert gate.update(True, 1.2, .2) is False
    assert gate.update(None, 1.3, .2) is None and gate.since is None
    assert gate.update(True, 1.4, .2) is False
    assert gate.update(True, 1.6, .2) is True
    gate.reset()
    assert gate.update(True, 1.7, .2) is False
    assert gate.update(True, 1.9, .2) is True


def test_a3_stationary_target_uses_visible_frame0_prior_with_age():
    frame = {'physical_time_sec': 2., 'roles': {'tray': {'centroid_world_m': [.1, .2, .3]}},
             'actual_initial_translation_m': {'tray': [99.,99.,99.]}}
    refs = {'time_sec': 0., 'surface_centroids_world_m': {'tray': [.095, .205, .3]}}
    priors = runner.scene_entry_grounding_priors(frame, refs, ['tray'])
    stage = {'target': 'tray', 'target_grounding': 'scene_entry_stationary_target'}
    delta, record = runner.stationary_target_prior(stage, priors=priors, planned_objects={'red_cube'},
                                                  manipulated_roles={'red_cube'}, physical_time=12.)
    assert np.allclose(delta, [.005, -.005, 0.])
    assert record['mode'] == 'scene_entry_stationary_target'
    assert record['age_sec'] == 10. and record['scene_entry_time_sec'] == 2.
    assert record['fresh_feedback'] is False and record['fallback_reason'] is None
    assert record['raw_visible_delta_m'] == delta.tolist()
    delta[:] = 100.
    assert priors['tray']['delta_m'][0] == pytest.approx(.005)


@pytest.mark.parametrize('mode,planned,manipulated,available,reason', [
    ('current_visible_pair', set(), set(), True, None),
    ('scene_entry_stationary_target', {'tray'}, set(), True, 'target_planned_for_manipulation'),
    ('scene_entry_stationary_target', set(), {'tray'}, True, 'target_already_manipulated'),
    ('scene_entry_stationary_target', set(), set(), False, 'initial_visible_pair_unavailable'),
])
def test_a3_target_prior_explicit_fallback(mode, planned, manipulated, available, reason):
    priors = {'tray': {'delta_m': [.004, .005, 0.], 'scene_entry_time_sec': 0., 'reference_time_sec': 0.}} if available else {}
    delta, record = runner.stationary_target_prior({'target': 'tray', 'target_grounding': mode},
        priors=priors, planned_objects=planned, manipulated_roles=manipulated, physical_time=10.)
    assert delta is None and record['mode'] == 'current_visible_pair'
    assert record['fallback_reason'] == reason and record['fresh_feedback'] is False


def test_a3_only_sorting_c02_c04_target_grounding_changes():
    plan = ROOT / 'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json'
    a2 = runner.GRAPH.compile_root_plan(plan, ROOT / 'evaluations/twin_rsi_v02/A2_OVERRIDES.json')
    a3 = runner.GRAPH.compile_root_plan(plan, ROOT / 'evaluations/twin_rsi_v02/A3_OVERRIDES.json')
    for task in TASKS:
        for previous, stage in zip(a2[task]['stages'], a3[task]['stages']):
            restored = dict(stage)
            if task == 'sorting_cube_bimanual' and stage['cycle_id'] in ('c02', 'c04'):
                assert restored.pop('target_grounding') == 'scene_entry_stationary_target'
            assert restored == previous
