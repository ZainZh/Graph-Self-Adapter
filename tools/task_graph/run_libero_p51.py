#!/usr/bin/env python3
"""Prepare or execute a frozen, single-reset P51 full LIBERO mixed neural/code engineering baseline.

Prepare does not construct a simulator. Run requires an existing binding and
creates a fresh output directory, records every state/RGB/action, and never retries.
"""
from __future__ import annotations
import argparse
import hashlib
from dataclasses import asdict
import json
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Set thread limits before importing numerical libraries.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from prove.task_graph.schema import TaskGraph
from prove.task_graph.grounding import GroundedSkill, RoleBinding, InputReference
from prove.task_graph.executor import GraphExecutor
from prove.task_graph.libero_demo_skill import FrozenDemoSkill
from prove.task_graph.libero_precision_p51 import MixedController, MixedRouter
import torch
torch.set_num_threads(1)
from tools.task_graph.collect_libero_demo_gt import prepare as prepare_environment, verify as verify_environment
from tools.task_graph.collect_libero_demo_gt import source_demo, snapshot, sha, dump


def prepare(source, graph, calibration, libero_root, robosuite_root, checkpoint):
    graph, calibration = Path(graph).resolve(), Path(calibration).resolve()
    task_graph = TaskGraph.load(graph)
    skill = FrozenDemoSkill.from_source(source, task_graph)
    checkpoint = Path(checkpoint).resolve()
    training_freeze = checkpoint.parent / 'TRAINING_FREEZE.json'
    package = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if package.get('training_freeze_sha256') != sha(training_freeze):
        raise ValueError('checkpoint/training-freeze mismatch')
    if set(package['paths']) != {'s3', 's6', 's7'}:
        raise ValueError('checkpoint precise-stage scope mismatch')
    model = MixedController(source, task_graph, checkpoint).model
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError('inference weights must be frozen')
    paths = [graph, calibration, checkpoint, training_freeze, Path(__file__).resolve()]
    paths += [ROOT / name for name in ('prove/__init__.py', 'prove/task_graph/__init__.py',
        'prove/task_graph/libero_precision_p51.py', 'prove/task_graph/libero_demo_skill.py',
        'prove/task_graph/schema.py', 'prove/task_graph/grounding.py', 'prove/task_graph/executor.py',
        'prove/task_graph/libero_supervisor.py', 'tools/__init__.py', 'tools/task_graph/__init__.py',
        'tools/task_graph/collect_libero_demo_gt.py', 'tools/task_graph/train_libero_precision_p51.py',
        'schemas/task_graph_v1.schema.json')]
    return dict(schema='prove.p51-libero-full-task-freeze.v1',
                mode='full_task_code_and_frozen_osc_mlp_nominal_demo_reference',
                environment=prepare_environment(source, libero_root, robosuite_root),
                graph=str(graph), calibration=str(calibration), checkpoint=str(checkpoint),
                training_freeze=str(training_freeze), training_freeze_sha256=sha(training_freeze),
                precise_stages=['s3','s6','s7'], reference='nominal_opened_demo_reference',
                controller=asdict(skill.config), implementation={str(p): sha(p) for p in paths},
                max_actions=700, initial_state_restores=1, automatic_retries=0,
                controller_packages={name: importlib.metadata.version(name) for name in ('scipy', 'jsonschema', 'torch')},
                adapter_executed=False, pretrained_policy_executed=False,
                live_action_inputs=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
                supervisor_authority='explicit_simulator_truth_only_for_stage_judgments')


def verify(binding):
    if binding.get('schema') != 'prove.p51-libero-full-task-freeze.v1':
        raise ValueError('unsupported qualifier binding')
    verify_environment(binding['environment'])
    for file, digest in binding['implementation'].items():
        if sha(file) != digest:
            raise ValueError(f'frozen code/config changed: {file}')
    env = binding['environment']
    if prepare(env['source'], binding['graph'], binding['calibration'], env['libero_root'], env['robosuite_root'], binding['checkpoint']) != binding:
        raise ValueError('binding content differs from source-derived configuration')


def grounding(graph, stage, instance_id, frame):
    return GroundedSkill.bind(stage,
        {key: RoleBinding(getattr(stage, key), getattr(stage, key) is not None)
         for key in ('actor', 'object', 'object_part', 'target')},
        (InputReference(f'frame:{frame}:robot_eef', 'robot_state', 'live observed robot proprioception'),),
        graph_id=graph.graph_id, task_id=graph.task_id, instance_id=instance_id)


def execute_episode(env, graph, skill, supervisor, output, max_actions=700, capture=snapshot):
    """Judge before action; completed stages rejudge the same physical observation."""
    output = Path(output)
    instance = output.name
    router = MixedRouter()
    for name in {stage.skill for stage in graph.nodes}:
        router.register(name, kind=router.required_kind(name), revision=skill.revision,
                        call=skill, compatible=lambda grounded: grounded.stage_id in skill.segments)
    executor = GraphExecutor(graph, router, instance_id=instance)
    obs = env._get_observations(force_update=True)
    actions = 0
    active = None
    final = None
    while True:
        arrays, state_supervisor = capture(env, obs)
        events = []
        skill.set_observation(obs)
        for _ in range(len(graph.nodes) + 1):
            stage = graph.stage(executor.current_stage)
            if active != stage.id:
                if active is not None:
                    skill.on_exit()
                skill.on_enter(stage.id)
                active = stage.id
            report = supervisor.observe(obs, stage.id)
            # Still evaluate final post-action state when physical budget expires.
            # A backend call is forbidden after budget exhaustion.
            if actions >= max_actions:
                predicates_complete = all(p.evaluate(report.facts) is True for p in stage.termination)
                if not predicates_complete:
                    final = dict(state='structural_failure', stage_id=stage.id,
                                 reason='episode_step_budget_exhausted', graph_complete=False,
                                 replan_requested=True)
                    events.append(dict(judge=final, supervisor=asdict(report)))
                    break
            result = executor.tick(grounding(graph, stage, instance, actions), report.facts, progress=report.progress)
            event = asdict(result)
            event['action'] = None if result.action is None else np.asarray(result.action).tolist()
            events.append(dict(judge=event, supervisor=asdict(report),
                               backend_kind=router.required_kind(stage.skill),
                               controller=dict(skill.diagnostics)))
            final = event
            if result.state == 'complete' and not result.graph_complete:
                continue
            break
        else:
            raise RuntimeError('unbounded same-observation graph transition')
        action = final.get('action')
        if action is None and not final.get('graph_complete') and final['state'] != 'structural_failure':
            final = dict(state='structural_failure', reason='missing_supervisor_measurement',
                         stage_id=executor.current_stage, graph_complete=False, replan_requested=True)
            events.append(dict(judge=final, explanation='No action issued; no new physical observation can resolve missing supervision.'))
        arrays['action_next'] = np.asarray(action if action is not None else [], dtype=float)
        frame_path = output / f'frame_{actions:04d}.npz'
        np.savez_compressed(frame_path, **arrays)
        row = dict(frame=actions, file=frame_path.name, sha256=sha(frame_path),
                   events=events, supervisor=state_supervisor)
        with (output / 'FRAMES.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if final.get('graph_complete') or final['state'] == 'structural_failure':
            break
        obs, _, _, _ = env.step(np.asarray(action, dtype=float))
        actions += 1
    skill.on_exit()
    return dict(schema='prove.p51-libero-full-task-result.v1',
                mode='full_task_code_and_frozen_osc_mlp_nominal_demo_reference', actions_executed=actions,
                frames=actions + 1, graph_complete=bool(final.get('graph_complete', False)),
                official_success=state_supervisor['success'], final_judgment=final,
                adapter_executed=False, independent_generalization_evidence=False)


def run(binding_path, output):
    binding = json.loads(Path(binding_path).read_text())
    verify(binding)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'BINDING.json', binding)
    env_binding = binding['environment']
    config_dir = output / 'libero_config'
    config_dir.mkdir()
    base = Path(env_binding['libero_root']) / 'libero/libero'
    dump(config_dir / 'config.yaml', dict(benchmark_root=str(base), bddl_files=str(base / 'bddl_files'),
         init_states=str(base / 'init_files'), datasets=str(Path(env_binding['source']).parent), assets=str(base / 'assets')))
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, env_binding['libero_root'])
    from libero.libero.envs import TASK_MAPPING
    from prove.task_graph.libero_supervisor import LiberoSupervisor
    env = None
    try:
        graph = TaskGraph.load(binding['graph'])
        skill = MixedController(env_binding['source'], graph, binding['checkpoint'])
        demo = source_demo(env_binding['source'])
        env = TASK_MAPPING[env_binding['env_args_original']['problem_name']](**env_binding['env_kwargs'])
        env.reset()
        env.reset_from_xml_string(env_binding['relocated_xml'])
        env.sim.reset()
        env.sim.set_state_from_flattened(demo['initial'])
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        names = {kind: [getattr(env.sim.model, kind + '_id2name')(i) for i in range(getattr(env.sim.model, count))]
                 for kind, count in (('body', 'nbody'), ('site', 'nsite'), ('geom', 'ngeom'), ('joint', 'njnt'))}
        dump(output / 'SUPERVISOR_NAMES.json', names)
        supervisor = LiberoSupervisor(env, json.loads(Path(binding['calibration']).read_text()), instance_id=output.name)
        before = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in skill.model.state_dict().values())).hexdigest()
        result = execute_episode(env, graph, skill, supervisor, output, binding['max_actions'])
        after = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in skill.model.state_dict().values())).hexdigest()
        result['neural_parameters_before'] = before
        result['neural_parameters_after'] = after
        result['frozen_neural_weights_unchanged'] = before == after
        if before != after:
            raise RuntimeError('neural action weights changed during inference')
        result['binding_sha256'] = sha(output / 'BINDING.json')
        dump(output / 'RESULT.json', result)
        return result
    except Exception as exc:
        dump(output / 'FAILURE.json', dict(error_type=type(exc).__name__, error=str(exc), automatic_retry=False))
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('prepare')
    for name in ('source', 'graph', 'calibration', 'libero-root', 'robosuite-root', 'checkpoint', 'output'):
        pre.add_argument('--' + name, type=Path, required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--binding', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists():
            raise FileExistsError(args.output)
        dump(args.output, prepare(args.source, args.graph, args.calibration, args.libero_root, args.robosuite_root, args.checkpoint))
    else:
        print(json.dumps(run(args.binding, args.output), indent=2))


if __name__ == '__main__':
    main()
