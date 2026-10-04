"""Frozen P47 representation/solver evaluation. No privileged facts before trace sealing.

Freeze schema ``prove.p47-evaluation.v1`` has:
  status='frozen', root_authorized=true, phase='development'|'independent_validation';
  steps_per_episode (10..240), execution.model_dir;
  sources={runtime source path: sha256}, files={model/PCA/vision path: sha256};
  robot_assets=source_assets(), engineering_gates={task: {path,sha256}};
  environments={task:{module,class,configuration_module,controller_module}};
  tasks={task: {models: {representation: {str(seed): {
    checkpoint:{path,sha256}, pca:{path,sha256} (or null for geometry)}}}}};
  cells=[{task,split,condition_seed,rep,model_seed,arm,video:boolean}].
The analytic reference is exactly rep='geometry', arm='multiview',
model_seed=null. It has no checkpoint, PCA, DINO call or duplicate model seed.
The fixed, continuous-gradient and direct arms share frozen P45 checkpoints.
Direct fitting minimizes the declared visible-ray quadratic on the same support.
Source and model/PCA bindings are exact. One frozen source snapshot is shared by
all workers. Each completed cell is retained even when other cells abort.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from prove.fast_adapter.geometry3d_adapter import GeometryLatent, pose
from prove.fast_adapter.p47_neural import ARMS, P47Latent, REPRESENTATIONS, SOLVER_CONTRACT, TASK_GEOMETRY, load_model
from prove.fast_adapter.p45_visual import FrozenP45Vision
from prove.fast_adapter.robot_geometry_env_p45 import source_assets
from tools.fast_adapter.p45_common import SOURCES as COMMON_SOURCES, js, make_env, observe_roles, sha

SOURCES = list(dict.fromkeys(COMMON_SOURCES + [
    'prove/fast_adapter/neural_geometry_adapter.py',
    'prove/fast_adapter/p45_neural.py',
    'prove/fast_adapter/p45_visual.py',
    'prove/fast_adapter/p47_neural.py',
    'tools/fast_adapter/p47_evaluate.py',
    'tools/fast_adapter/p47_prepare.py',
    'tools/fast_adapter/p47_aggregate.py',
    'tools/fast_adapter/p45_aggregate.py',
    'tools/fast_adapter/p45_evaluate.py',
    'tools/fast_adapter/p42_stats.py',
    'tools/fast_adapter/p45_batch_worker.py',
]))
VISION_FILES = [
    'data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt',
    'data/sources/lila-wam-reference-20260923/models/vla_model_fm.py',
]
PUBLIC_KEYS = ('corners', 'camera_position', 'camera_rotation', 'K', 'position',
               'rotation', 'role_bbox', 'role_visible')


def parameters_sha(model):
    if model is None:
        return None
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def binding(path):
    return dict(path=str(path), sha256=sha(path))


def check_binding(item, files=None):
    if not isinstance(item, dict) or not {'path', 'sha256'} <= set(item):
        raise ValueError('Expected an explicit path/hash artifact binding')
    if files is not None and files.get(item['path']) != item['sha256']:
        raise ValueError(f'Artifact is not included in the frozen file inventory: {item["path"]}')
    if sha(item['path']) != item['sha256']:
        raise ValueError(f'Frozen artifact changed: {item["path"]}')


def validate_sources(frozen):
    needed = set(SOURCES)
    for backend in frozen['environments'].values():
        if set(backend) != {'module', 'class', 'configuration_module', 'controller_module'}:
            raise ValueError('Each task must declare its exact environment/controller modules')
        for field in ('module', 'configuration_module', 'controller_module'):
            needed.add(backend[field].replace('.', '/') + '.py')
    if not needed <= set(frozen['sources']):
        raise ValueError('Runtime source closure is incomplete')
    for filename, digest in frozen['sources'].items():
        if sha(filename) != digest:
            raise ValueError(f'Frozen source changed: {filename}')


def model_binding(frozen, cell):
    if cell['arm'] == 'multiview':
        return dict(method='analytic_visible_multiview', checkpoint=None, pca=None)
    return frozen['tasks'][cell['task']]['models'][cell['rep']][str(cell['model_seed'])]


class MultiviewReference:
    """Read-only reuse of P41 fitting, with P45 logging and task identity.

    The fit sees the same public corners, calibration and actual camera motion.
    It preserves the first 12 and latest 20 views after the common 32-view cap,
    exactly as the neural adapter. No extra observation or hidden pose is used.
    """
    def __init__(self, task):
        self.task = task
        self.geometry = GeometryLatent(TASK_GEOMETRY[task], 'multiview')
        self.records = []
        self.update_events = 0  # No neural latent gradient events.
        self.fit_events = 0

    def update(self, obs, t):
        if obs['corners'] is None:
            return
        previous = None if self.geometry.points is None else self.geometry.points.copy()
        self.geometry.update(obs, t)
        points = self.geometry.points
        if previous is None:
            previous = points.copy()
        fitted = len(self.geometry.history) >= 3
        self.fit_events += int(fitted)
        self.records.append(dict(t=int(t), S=min(len(self.geometry.history), 32), U=0,
            fit_updated=fitted, fit_events=self.fit_events, latent_change=0.,
            point_change=float(np.linalg.norm(points - previous)),
            pre_update_points=previous.tolist(), fitted_points=points.tolist(),
            method='analytic_visible_multiview'))

    def estimate(self):
        return self.geometry.estimate()


def assets_for_environment(backend):
    module = importlib.import_module(backend['controller_module'])
    return getattr(module, 'source_assets', source_assets)()


TASKS = ('cup', 'insert', 'slider')
CAMPAIGN = 'p47_representation_solver'
MAIN_BASE = {'cup': 480100, 'insert': 481100, 'slider': 482100}
DEVELOPMENT_BASE = {'cup': 455406, 'insert': 456406, 'slider': 457406}


def make_cells(task, phase):
    """Exactly 13 physical cells per condition; one model-free reference."""
    if task not in TASKS or phase not in ('development', 'independent_validation'):
        raise ValueError('Unknown P47 task or phase')
    start = DEVELOPMENT_BASE[task] if phase == 'development' else MAIN_BASE[task]
    count = 2 if phase == 'development' else 32
    cells = [dict(task=task, split='geometry', condition_seed=start + index,
                  rep=rep, model_seed=seed, arm=arm,
                  video=rep == 'relative_fusion' and seed == 11 and index == 0)
             for rep in REPRESENTATIONS for seed in (11, 22)
             for index in range(count) for arm in ARMS]
    cells.extend(dict(task=task, split='geometry', condition_seed=start + index,
                      rep='geometry', model_seed=None, arm='multiview', video=index == 0)
                 for index in range(count))
    return cells


def validate_development_eligibility(item, frozen):
    """Require the root's comparison-validity review, never task-success admission."""
    check_binding(item, frozen['files'])
    record = json.loads(Path(item['path']).read_text())
    if (record.get('schema') != 'prove.p47-development-eligibility.v1'
            or record.get('status') != 'complete' or record.get('root_reviewed') is not True
            or record.get('passed') is not True or record.get('prespecified_tasks') != list(TASKS)
            or record.get('success_rate_inclusion_gate') is not False
            or record.get('plan') != frozen['plan']
            or record.get('analysis_contract_file') != frozen['analysis_contract_file']
            or any(record.get('checks', {}).get(key) is not True for key in
                   ('integrity_valid', 'numerical_valid', 'physical_comparison_valid'))
            or set(record.get('development', {})) != set(TASKS)):
        raise ValueError('P47 requires a complete root review of development comparison validity')
    for task, review in record['development'].items():
        check_binding(review['evaluation_freeze'])
        check_binding(review['evidence'])
        dev = json.loads(Path(review['evaluation_freeze']['path']).read_text())
        if (dev.get('schema') != 'prove.p47-evaluation.v1' or dev.get('phase') != 'development'
                or dev.get('campaign') != CAMPAIGN or dev.get('status') != 'frozen'
                or dev.get('root_authorized') is not True
                or dev.get('cells') != make_cells(task, 'development')
                or dev.get('plan') != frozen['plan']
                or dev.get('analysis_contract_file') != frozen['analysis_contract_file']
                or dev.get('solver_contract') != frozen['solver_contract']):
            raise ValueError('Root review development scope or candidate contract mismatch')
        if task in frozen['tasks']:
            for key in ('tasks', 'environments', 'engineering_gates', 'robot_assets',
                        'steps_per_episode', 'execution', 'sources'):
                if dev[key] != frozen[key]:
                    raise ValueError('Independent validation changed the reviewed development candidate')
    return record


def validate_campaign(frozen):
    if (frozen.get('campaign') != CAMPAIGN or frozen.get('prespecified_tasks') != list(TASKS)
            or len(frozen.get('tasks', {})) != 1
            or frozen.get('solver_contract') != SOLVER_CONTRACT):
        raise ValueError('P47 campaign, task scope or solver contract mismatch')
    task = next(iter(frozen['tasks']))
    models = frozen['tasks'][task].get('models', {})
    if (set(models) != set(REPRESENTATIONS)
            or any(set(models[rep]) != {'11', '22'} for rep in REPRESENTATIONS)
            or frozen.get('cells') != make_cells(task, frozen['phase'])):
        raise ValueError('P47 requires its exact G/R, seed11/22, three-solver OOD matrix')
    for key in ('plan', 'analysis_contract_file', 'source_p46_freeze', 'source_p45_development'):
        check_binding(frozen[key], frozen['files'])
    if json.loads(Path(frozen['analysis_contract_file']['path']).read_text()) != frozen['analysis_contract']:
        raise ValueError('Embedded P47 analysis contract differs from its bound file')
    if frozen['phase'] == 'independent_validation':
        if not frozen.get('development_eligibility'):
            raise ValueError('Independent validation requires root DEVELOPMENT_ELIGIBILITY')
        validate_development_eligibility(frozen['development_eligibility'], frozen)
    elif 'development_eligibility' in frozen:
        raise ValueError('Development cannot inherit a later validation admission')


def validate_freeze(path):
    frozen = json.loads(Path(path).read_text())
    if (frozen.get('schema') != 'prove.p47-evaluation.v1'
            or frozen.get('status') != 'frozen' or frozen.get('root_authorized') is not True
            or frozen.get('phase') not in ('development', 'independent_validation')):
        raise ValueError('A root-authorized result-free P47 evaluation freeze is required')
    validate_campaign(frozen)
    if not isinstance(frozen.get('steps_per_episode'), int) or not 10 <= frozen['steps_per_episode'] <= 240:
        raise ValueError('P47 evaluation requires one common 10..240 step budget')
    validate_sources(frozen)
    if not frozen.get('robot_assets') or any(assets_for_environment(backend) != frozen['robot_assets']
                                            for backend in frozen['environments'].values()):
        raise ValueError('Frozen robot XML/mesh assets changed or are missing')
    for filename, digest in frozen['files'].items():
        if sha(filename) != digest:
            raise ValueError(f'Frozen file changed: {filename}')
    if not frozen.get('cells'):
        raise ValueError('An exact, nonempty evaluation cell list is required')
    identifiers = set()
    for cell in frozen['cells']:
        required = {'task', 'split', 'condition_seed', 'rep', 'model_seed', 'arm', 'video'}
        if set(cell) != required:
            raise ValueError('Unexpected or missing evaluation cell fields')
        reference = cell['arm'] == 'multiview'
        if (cell['task'] not in TASK_GEOMETRY or cell['split'] != 'geometry'
                or cell['rep'] not in REPRESENTATIONS or cell['arm'] not in (*ARMS, 'multiview')
                or (not reference and cell['model_seed'] not in (11, 22))
                or (reference and (cell['model_seed'] is not None or cell['rep'] != 'geometry'))
                or type(cell['video']) is not bool
                or type(cell['condition_seed']) is not int):
            raise ValueError('Unsupported P47 evaluation cell')
        identifier = tuple(cell[key] for key in ('task', 'split', 'condition_seed', 'rep', 'model_seed', 'arm'))
        if identifier in identifiers:
            raise ValueError('Duplicate physical evaluation cell')
        identifiers.add(identifier)
        if cell['task'] not in frozen['environments']:
            raise ValueError('A task has no frozen environment binding')
        candidate = model_binding(frozen, cell)
        for key in (() if reference else ('checkpoint', 'pca')):
            if key == 'pca' and cell['rep'] == 'geometry' and candidate.get(key) is None:
                continue
            item = candidate[key]
            if frozen['files'].get(item['path']) != item['sha256']:
                raise ValueError(f'Candidate {key} is absent from frozen file inventory')
    used_tasks = {cell['task'] for cell in frozen['cells']}
    for task in used_tasks:
        gate = frozen['engineering_gates'][task]
        check_binding(gate)
        record = json.loads(Path(gate['path']).read_text())
        if 'tasks' in record:
            record = record['tasks'].get(task, {})
        if record.get('task', task) != task or not (record.get('passed') is True or record.get('qualified') is True):
            raise ValueError(f'Engineering qualification has not passed for {task}')
        if record.get('environment') != frozen['environments'][task]:
            raise ValueError(f'Engineering gate/backend version mismatch for {task}')
    if any(cell['rep'] != 'geometry' for cell in frozen['cells']):
        vision = VISION_FILES + [str(Path(frozen['execution']['model_dir']) / name)
                                 for name in ('model.safetensors', 'config.json', 'preprocessor_config.json')]
        if not set(vision) <= set(frozen['files']):
            raise ValueError('Frozen DINO/LiLa assets missing from file inventory')
    return frozen


def snapshot_sources(freeze_path, frozen):
    """Atomic create-once copies tolerate concurrent workers sharing a freeze."""
    root = Path(freeze_path).parent / 'source_snapshot' / sha(freeze_path)
    for source, digest in frozen['sources'].items():
        relative = Path(source)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Frozen source paths must be repository-relative')
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(Path(source).read_bytes())
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, dest)
                except FileExistsError:
                    pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if sha(dest) != digest:
            raise ValueError('Frozen source snapshot differs from its source inventory')
    return str(root)


def snapshot(env):
    state = env.snapshot()
    state_spec = env.mj.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(env.mj.mj_stateSize(env.model, state_spec))
    env.mj.mj_getState(env.model, env.data, integration, state_spec)
    return dict(state, integration=integration)


def restore(env, state):
    env.restore(state)
    env.mj.mj_setState(env.model, env.data, state['integration'], env.mj.mjtState.mjSTATE_INTEGRATION)
    env.mj.mj_forward(env.model, env.data)


def counterfactual_command(controller, t, estimate, proprio):
    before = controller.state_dict()
    try:
        return controller.command(t, estimate, proprio)
    finally:
        controller.load_state_dict(before)


def evaluate_sealed(env, traces, terminal, out):
    out = Path(out)
    seal = json.loads((out / 'SEAL.json').read_text())
    for filename in ('TRACE.npy', 'TERMINAL.npy', 'OBS.npy', 'PRE_RUN.json'):
        if seal['files'].get(filename) != sha(out / filename):
            raise ValueError('Evaluation attempted before a valid trajectory seal')
    restore(env, terminal)
    facts = dict(env.facts())
    held, penetrations = [], []
    for state in [row['state'] for row in traces[-9:]] + [terminal]:
        restore(env, state)
        held.append(bool(env.facts()['success']))
        distances = np.asarray([float(contact.dist) for contact in env.data.contact])
        penetrations.append(float(np.max(np.maximum(0., -distances), initial=0.)))
    if len(held) != 10:
        raise ValueError('A sustained success requires exactly ten sealed terminal states')
    arm = np.asarray([row['robot_tracking']['max_substep_arm_limit_violation_rad'] for row in traces])
    finger = np.asarray([row['robot_tracking']['max_substep_finger_limit_violation_m'] for row in traces])
    penetration = np.asarray([row['robot_tracking']['max_substep_penetration_m'] for row in traces] + penetrations)
    states = [row['state'] for row in traces] + [terminal]
    finite = bool(all(np.isfinite(values).all() for values in (arm, finger, penetration))
                  and all(np.isfinite(state.get(key, [])).all() for state in states
                          for key in ('qpos', 'qvel', 'integration', 'act', 'ctrl', 'qacc_warmstart'))
                  and all(np.isfinite(row.get(key, [])).all() for row in traces
                          for key in ('command_p', 'command_R')))
    # Nonfinite values cannot disappear through Python max's NaN ordering.
    max_arm = float(np.max(arm)) if np.isfinite(arm).all() else None
    max_finger = float(np.max(finger)) if np.isfinite(finger).all() else None
    max_penetration = float(np.max(penetration)) if np.isfinite(penetration).all() else None
    success = bool(all(held) and np.isfinite(penetrations).all() and max(penetrations) < .002)
    mechanical_valid = bool(finite and max_penetration < .002 and max_arm <= .001 and max_finger <= .0001)
    validated_success = bool(success and mechanical_valid)
    facts.update(sustained_native_success=bool(all(held)), success=success,
                 last10_successes=sum(held),
                 max_last10_penetration=max(penetrations) if np.isfinite(penetrations).all() else None,
                 max_arm_joint_limit_violation_rad=max_arm,
                 max_finger_joint_limit_violation_m=max_finger,
                 max_full_trajectory_contact_penetration=max_penetration,
                 trajectory_finite=finite, mechanical_valid=mechanical_valid,
                 validated_success=validated_success, physical_qualified=validated_success,
                 outcome_class='validated_success' if validated_success else
                               'mechanical_anomaly' if not mechanical_valid else 'ordinary_task_failure',
                 task_success_denominator_included=True)
    restore(env, terminal)
    return facts


def rollout(cell_index, cell, frozen, freeze_binding, source_snapshot, out, model, vision):
    validate_sources(frozen)
    backend = frozen['environments'][cell['task']]
    if assets_for_environment(backend) != frozen['robot_assets']:
        raise ValueError('Robot assets changed before cell execution')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    candidate = model_binding(frozen, cell)
    js(out / 'PRE_RUN.json', dict(
        cell_index=cell_index, cell=cell, phase=frozen['phase'], steps=frozen['steps_per_episode'],
        freeze=freeze_binding, candidate=candidate, source_snapshot=source_snapshot,
        environment=backend,
        engineering_gate=frozen['engineering_gates'][cell['task']], timestamp=time.time(),
        solver_contract=frozen['solver_contract'],
        input_contract='Visible RGB/masks/corners, camera calibration, own tool state; no target truth',
        latent_contract=('Visible-ray multiview fitting only; retain final estimate once controller locks moving target'
                         if cell['arm'] == 'multiview' else
                         'Only z changes online; retain final estimate once controller locks moving target'),
        insertion_secure_grasp_assumed=cell['task'] == 'insert',
    ))
    env = None
    traces, observations, frames, details = [], [], [], []
    before_hash = parameters_sha(model)
    started = time.monotonic()
    try:
        env = make_env(cell['task'], cell['condition_seed'], cell['split'], backend=backend)
        if env.model.nmocap != 0 or env.model.nu < 7:
            raise ValueError('P47 requires an actuated articulated robot without mocap control')
        if cell['task'] == 'insert':
            env.calibrate_held_geometry()
        length = env.tool_calibration.get('length', 0.)
        from tools.fast_adapter.p45_common import make_controller
        controller = make_controller(env, length, backend)
        adapter = MultiviewReference(cell['task']) if cell['arm'] == 'multiview' else P47Latent(
            model, cell['arm'])
        p, rotation = env.proprio()
        previous_action = np.r_[p, rotation.ravel(), float(cell['task'] == 'insert')]
        for t in range(frozen['steps_per_episode']):
            observed, rgb, masks = observe_roles(env)
            obs = {key: copy.deepcopy(observed[key]) for key in PUBLIC_KEYS}
            obs['local'] = np.zeros(48, np.float32)
            if obs['corners'] is not None and not controller.adaptation_frozen:
                if cell['rep'] != 'geometry':
                    obs['local'] = vision.encode(rgb[None], masks[None]).cpu().numpy()[0]
                adapter.update(obs, t)
            observations.append(dict(obs, t=t, previous_action=previous_action.copy(),
                                     adaptation_frozen=controller.adaptation_frozen))
            # The last estimate persists after closure; a moving object must
            # not be re-fit as if the whole historical target were stationary.
            estimate = adapter.estimate()
            proprio = env.proprio()
            before = controller.state_dict()
            update = copy.deepcopy(adapter.records[-1]) if adapter.records and adapter.records[-1]['t'] == t else None
            counterfactual = None
            if update is not None:
                prior = pose(np.asarray(update['pre_update_points']))
                counterfactual = counterfactual_command(controller, t, prior, proprio)
            p, rotation, closed, stage = controller.command(t, estimate, proprio)
            after = controller.state_dict()
            if update is not None:
                p0, r0, closed0, stage0 = counterfactual
                update.update(same_state_action_position_delta=float(np.linalg.norm(p - p0)),
                              same_state_action_rotation_delta=float(np.linalg.norm(
                                  Rotation.from_matrix(rotation @ r0.T).as_rotvec())),
                              same_state_closed_changed=bool(closed != closed0),
                              same_state_stage_changed=bool(stage != stage0))
            row = dict(t=t, state=snapshot(env), command_p=p.copy(), command_R=rotation.copy(),
                       closed=bool(closed), stage=stage, estimate=copy.deepcopy(estimate),
                       proprio=proprio, controller_before=before, controller_after=after,
                       adaptation=update, controls_before=env.data.ctrl.copy())
            if cell['video']:
                frames.append(np.concatenate([env.render_global(), rgb], axis=1))
                details.append(env.render_detail())
            env.step(p, rotation, closed)
            row['controls_after'] = env.data.ctrl.copy()
            row['robot_tracking'] = env.robot_tracking_diagnostics()
            traces.append(row)
            previous_action = np.r_[p, rotation.ravel(), float(closed)]
        terminal = snapshot(env)
        if cell['video']:
            _, rgb, _ = observe_roles(env)
            frames.append(np.concatenate([env.render_global(), rgb], axis=1))
            details.append(env.render_detail())
        np.save(out / 'TRACE.npy', np.array(traces, dtype=object), allow_pickle=True)
        np.save(out / 'OBS.npy', np.array(observations, dtype=object), allow_pickle=True)
        np.save(out / 'TERMINAL.npy', terminal, allow_pickle=True)
        filenames = ['TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json']
        if cell['video']:
            np.savez_compressed(out / 'FRAMES.npz', frames=np.array(frames))
            np.savez_compressed(out / 'DETAIL_FRAMES.npz', frames=np.array(details))
            filenames += ['FRAMES.npz', 'DETAIL_FRAMES.npz']
        after_hash = parameters_sha(model)
        js(out / 'SEAL.json', dict(schema='prove.p45-trajectory-seal.v1',
           campaign=CAMPAIGN, freeze=freeze_binding,
           files={filename: sha(out / filename) for filename in filenames},
           neural_parameters_before=before_hash, neural_parameters_after=after_hash))
        if before_hash != after_hash:
            raise RuntimeError('Runtime changed neural weights; evaluation is invalid')
        facts = evaluate_sealed(env, traces, terminal, out)
        changes = [row['adaptation'] for row in traces if row['adaptation'] is not None]
        result = dict(
            **cell, cell_index=cell_index, phase=frozen['phase'], **facts,
            steps=frozen['steps_per_episode'], neural_weights_unchanged=True if model is not None else None,
            adaptation_updates=sum(row['U'] for row in changes),
            gradient_steps=sum(row['U'] for row in changes) if cell['arm'] == 'continuous' else 0,
            direct_solves=adapter.update_events if cell['arm'] == 'direct' else 0,
            adaptation_events=adapter.update_events,
            geometric_fit_events=getattr(adapter, 'fit_events', 0),
            valid_frames=sum(row['corners'] is not None for row in observations),
            max_latent_change=max([row['latent_change'] for row in changes] + [0.]),
            max_same_state_action_position_delta=max([row['same_state_action_position_delta'] for row in changes] + [0.]),
            max_same_state_action_rotation_delta=max([row['same_state_action_rotation_delta'] for row in changes] + [0.]),
            final_controller_phase=controller.phase, adaptation_frozen=controller.adaptation_frozen,
            max_position_tracking_error=max([row['robot_tracking']['position_error'] for row in traces] + [0.]),
            max_angle_tracking_error=max([row['robot_tracking']['angle_error'] for row in traces] + [0.]),
            tool_calibration={key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in env.tool_calibration.items()},
            elapsed_seconds=time.monotonic() - started,
        )
        js(out / 'RESULT.json', result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        partial = {}
        for filename, records in [('TRACE_PARTIAL.npy', traces), ('OBS_PARTIAL.npy', observations)]:
            np.save(out / filename, np.array(records, dtype=object), allow_pickle=True)
            partial[filename] = sha(out / filename)
        if env is not None:
            try:
                np.save(out / 'STATE_AT_ABORT.npy', snapshot(env), allow_pickle=True)
                partial['STATE_AT_ABORT.npy'] = sha(out / 'STATE_AT_ABORT.npy')
            except Exception:
                pass
        js(out / 'ABORT.json', dict(cell=cell, type=type(exc).__name__, reason=str(exc),
           completed_steps=len(traces), partial_files=partial,
           solver_diagnostics=getattr(exc, 'diagnostics', None),
           interpretation='Execution/implementation exception retained; not silently removed or relabeled as model failure'))
        raise
    finally:
        if env is not None:
            env.close()


class WorkerModels:
    """One frozen DINO/LiLa instance per worker; tiny models/PCA are cached."""
    def __init__(self, frozen, device):
        self.frozen, self.device = frozen, device
        self.models, self.visions = {}, {}
        self.packages = {}
        self.raw_vision = None

    def get(self, cell):
        if cell['arm'] == 'multiview':
            return None, None
        candidate = model_binding(self.frozen, cell)
        checkpoint = candidate['checkpoint']
        if checkpoint['sha256'] not in self.models:
            check_binding(checkpoint, self.frozen['files'])
            model, package = load_model(checkpoint['path'], 'cpu')
            self.models[checkpoint['sha256']] = model
            self.packages[checkpoint['sha256']] = package
        model = self.models[checkpoint['sha256']]
        package = self.packages[checkpoint['sha256']]
        if (model.task != cell['task'] or model.representation != cell['rep']
                or package['seed'] != cell['model_seed']):
            raise ValueError('Checkpoint task/representation/model seed mismatch')
        if model.width != 96 or model.latent_dim != 8 or package['updates'] != 2000:
            raise ValueError('Checkpoint differs from the fixed P45 training/parameter budget')
        vision = None
        if cell['rep'] != 'geometry':
            pca = candidate['pca']
            if package['pca'][cell['rep']]['sha256'] != pca['sha256']:
                raise ValueError('Runtime projection differs from the checkpoint training projection')
            if pca['sha256'] not in self.visions:
                check_binding(pca, self.frozen['files'])
                vision = FrozenP45Vision(pca['path'], cell['rep'],
                    model_dir=self.frozen['execution']['model_dir'], device=self.device,
                    microbatch=1, raw_vision=self.raw_vision)
                self.raw_vision = vision.raw_vision
                self.visions[pca['sha256']] = vision
            vision = self.visions[pca['sha256']]
            if vision.package['representation'] != cell['rep']:
                raise ValueError('Cached projection/representation mismatch')
        return model, vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cell-index', type=int)
    group.add_argument('--cell-indices', help='Comma-separated exact frozen indices for one worker')
    parser.add_argument('--out', required=True, help='Worker-shared output root; each cell gets cell_NNNNNN')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    frozen = validate_freeze(args.freeze)
    indices = [args.cell_index] if args.cell_index is not None else [int(value) for value in args.cell_indices.split(',')]
    if not indices or len(indices) != len(set(indices)) or any(index < 0 or index >= len(frozen['cells']) for index in indices):
        raise ValueError('Requested cell indices are not a unique subset of the frozen scope')
    source_snapshot = snapshot_sources(args.freeze, frozen)
    freeze_binding = binding(args.freeze)
    cache = WorkerModels(frozen, args.device)
    failures = []
    for index in indices:
        cell = frozen['cells'][index]
        out = Path(args.out) / f'cell_{index:06d}'
        try:
            model, vision = cache.get(cell)
            rollout(index, cell, frozen, freeze_binding, source_snapshot, out, model, vision)
        except Exception as exc:
            failures.append(dict(cell_index=index, type=type(exc).__name__, reason=str(exc)))
            # Admission/source errors cannot create an unsafe rollout. Record
            # them alongside the worker summary without overwriting prior cells.
            print(json.dumps(dict(cell_index=index, execution_error=str(exc))), flush=True)
    record = dict(freeze=freeze_binding, requested_indices=indices, failures=failures,
                  completed=len(indices) - len(failures), failed=len(failures))
    summary = Path(args.out) / f'WORKER_{os.getpid()}_{time.time_ns()}.json'
    js(summary, record)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
