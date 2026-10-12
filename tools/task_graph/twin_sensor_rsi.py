#!/usr/bin/env python3
"""Sensor-only additive facade over immutable v02 task controllers.

No hardware transport. Execution requires an independently reviewed v03 freeze.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from prove.task_graph.twin_sensor_judge import JudgeConfig
from prove.task_graph.twin_sensor_adapter import sensor_world_class, sensor_holding_contract

TASK_CANDIDATES = {'put_knife_board_new': 'A2', 'put_mug_on_box': 'A2', 'sorting_cube_bimanual': 'B0'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_runner(task):
    candidate = TASK_CANDIDATES[task]
    prefix = REPO / 'runs/twin_rsi_v02'
    snapshot = prefix / 'source_snapshot' if candidate == 'B0' else prefix / candidate / 'source_snapshot'
    path = snapshot / 'tools/task_graph/twin_rsi.py'
    spec = importlib.util.spec_from_file_location('_sensor_parent_' + candidate, path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner, snapshot, REPO / f'task_packages/twin_rsi_v02/{candidate}/packages'


def install_sensor_proxy(runner, *, task, packages, config, mode, output, timeout_sec, provenance=None):
    solution = packages / ('twin_' + task) / 'solution'
    references = json.loads((solution / 'refs/visible_centroids.json').read_text())
    physics = runner._physics()
    World = sensor_world_class(physics, config=config, references=references,
                              judge_mode=mode, seal_path=output / 'SENSOR_PREDICTIONS_SEALED.json',
                              timeout_sec=timeout_sec, provenance=provenance)
    worlds = []
    class RegisteredWorld(World):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            worlds.append(self)
    proxy = types.SimpleNamespace(**{key: value for key, value in vars(physics).items() if not key.startswith('__')})
    proxy.TwinWorld = RegisteredWorld
    runner._physics = lambda: proxy
    if mode == 'sensor_only':
        # Preserve UNKNOWN in sensor OR contracts rather than bool(None) -> False.
        runner.holding_fact = lambda contract, raw, **kwargs: sensor_holding_contract(
            {'holding_contract': contract, 'target': kwargs.get('target_role'), 'goal_relation': kwargs.get('goal_relation')}, raw)
    OriginalRuntime = runner.GRAPH.TwinGraphRuntime
    class SensorRuntime(OriginalRuntime):
        @property
        def stage(self):
            stage = super().stage
            if worlds:
                world = worlds[-1]
                if world.sensor_stage is not None and world.sensor_stage['stage_id'] != stage['stage_id']:
                    world.sensor_unknown_since = None
                world.sensor_stage, world.sensor_runtime = stage, self
            return stage

        def transition(self, facts):
            if worlds and worlds[-1].sensor_motion_blocked:
                return False
            return super().transition(facts)
    runner.GRAPH.TwinGraphRuntime = SensorRuntime
    original_dump = runner.dump_json
    def dump(path, report):
        if report.get('schema') == 'graph-self-adapter.twin-run.v1':
            world = worlds[-1]
            report['sensor_backend'] = {'schema': 'graph-self-adapter.sensor-backend.v1',
                'judge_mode': mode, 'servo': 'raw_proprio_only_v1', 'controller_default': 'fixed_z',
                'mask_source': 'simulator_GT_semantic_mask_proxy; SAM_not_integrated_in_Graph',
                'safety_evaluator': 'separate_privileged_original_thresholds_unchanged',
                'prediction_seal': {'path': str(output / 'SENSOR_PREDICTIONS_SEALED.json'),
                                    'sha256': world.sensor_prediction_sha256},
                'hardware_qualified': False, 'actual_runtime_provenance': provenance or {},
                'execution_variant': 'sensor_only_judge' if mode == 'sensor_only' else 'explicit_gt_judge_control'}
        return original_dump(path, report)
    runner.dump_json = dump
    return worlds


def validate_execution(scope_path, freeze_path):
    scope, freeze = json.loads(scope_path.read_text()), json.loads(freeze_path.read_text())
    if scope.get('execution_authorized') is not True:
        raise ValueError('v03 execution is not authorized by the reviewed scope')
    if freeze.get('schema') != 'graph-self-adapter.sensor-execution-freeze.v1':
        raise ValueError('unsupported sensor execution freeze')
    if freeze.get('scope_sha256') != digest(scope_path):
        raise ValueError('sensor execution scope hash mismatch')
    bound = set()
    for binding in freeze['bindings']:
        path = Path(binding['path']).resolve()
        if digest(path) != binding['sha256']:
            raise ValueError('sensor execution binding mismatch: ' + str(path))
        bound.add(path)
    for relative in ('tools/task_graph/twin_sensor_rsi.py', 'prove/task_graph/twin_sensor_judge.py',
                     'prove/task_graph/twin_sensor_adapter.py'):
        if (REPO / relative).resolve() not in bound:
            raise ValueError('missing sensor source binding: ' + relative)
    freeze["_validated_bound_paths"] = bound
    return freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_CANDIDATES, required=True)
    parser.add_argument('--judge-mode', choices=('sensor_only', 'privileged_gt_control'), default='sensor_only')
    parser.add_argument('--config', type=Path, default=REPO / 'evaluations/twin_sensor_v03/SENSOR_CONFIG_PROPOSED.json')
    parser.add_argument('--contract', type=Path, default=REPO / 'evaluations/twin_sensor_v03/SENSOR_CONTRACT_PROPOSED.json')
    parser.add_argument('--scope', type=Path, default=REPO / 'evaluations/twin_sensor_v03/SCOPE.json')
    parser.add_argument('--execution-freeze', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trial', type=int, choices=range(5), default=0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('fresh output required')
    cfg = json.loads(args.config.read_text())
    config = JudgeConfig(**cfg['thresholds'])
    runner, snapshot, packages = load_frozen_runner(args.task)
    solution = packages / ('twin_' + args.task) / 'solution'
    runner.verify_freeze(solution)
    freeze = json.loads((solution / 'CANDIDATE_FREEZE.json').read_text())
    ready = {'task': args.task, 'candidate': TASK_CANDIDATES[args.task], 'judge_mode': args.judge_mode,
             'fixed_z': True, 'dry_run': not args.execute, 'source_snapshot': str(snapshot),
             'graph_sha256': digest(solution / 'graph/task_graph.json'),
             'model_sha256': digest(solution / 'models/local_skill.npz'), 'output': str(args.output)}
    if not args.execute:
        print(json.dumps(ready, indent=2)); return ready
    if args.execution_freeze is None:
        raise ValueError('reviewed result-free execution freeze required; no rollout started')
    execution = validate_execution(args.scope, args.execution_freeze)
    matches = [job for job in execution['jobs']
        if job['task'] == args.task and job['trial'] == args.trial and job['judge_mode'] == args.judge_mode
        and Path(job['output']).resolve() == args.output.resolve()]
    if len(matches) != 1:
        raise ValueError('unique exact task/trial/mode/output job absent from frozen campaign')
    actual_paths = [Path(__file__).resolve(), REPO / 'prove/task_graph/twin_sensor_judge.py',
        REPO / 'prove/task_graph/twin_sensor_adapter.py', args.config.resolve(), args.scope.resolve(),
        args.contract.resolve()]
    if not set(actual_paths).issubset(execution['_validated_bound_paths']):
        raise ValueError('actual config/contract/scope is missing from execution bindings')
    provenance = {str(path): digest(path) for path in actual_paths}
    provenance['execution_freeze_sha256'] = digest(args.execution_freeze)
    install_sensor_proxy(runner, task=args.task, packages=packages, config=config, mode=args.judge_mode,
                         output=args.output, timeout_sec=cfg['unknown_holding_timeout_sec'], provenance=provenance)
    run_args = runner.parser().parse_args([
        '--plan', str(REPO / 'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json'),
        '--assets', str(REPO / 'evaluations/twin_rsi_v02/EXTERNAL_ASSET_BINDINGS.json'),
        '--packages', str(packages), 'run', '--task', args.task,
        '--candidate-id', freeze['candidate_id'], '--condition-set', 'development_v02',
        '--condition-seed', '51005', '--trial', str(args.trial),
        '--runtime-contract', str(REPO / 'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json'),
        '--output', str(args.output)])
    result = runner.run(run_args)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
