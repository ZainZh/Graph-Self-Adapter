"""P44 v05 post-repair frozen-neural cup recheck on an articulated Panda.

This runner is development only. Its actor gets visible images/masks, calibrated
camera/tool poses and its own previous latent, never target geometry or success.
The physical state/action/observation records are sealed before evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from prove.fast_adapter.geometry3d_env import configuration
from prove.fast_adapter.geometry3d_adapter import pose
from prove.fast_adapter.neural_geometry_adapter import NeuralGeometryLatent, load_model
from prove.fast_adapter.neural_geometry_visual import FrozenRoleVision, mask_geometry


SOURCE = [
    'prove/fast_adapter/geometry3d_env.py',
    'prove/fast_adapter/geometry3d_adapter.py',
    'prove/fast_adapter/robot_geometry_env_v05.py',
    'prove/fast_adapter/neural_geometry_adapter.py',
    'prove/fast_adapter/neural_geometry_visual.py',
    'prove/fast_adapter/lila_frontend.py',
    'prove/fast_adapter/visual_alignment.py',
    'tools/fast_adapter/p44_evaluate_v05.py',
]
SCOPE = 'development_embodiment_repair_recheck'
VISION_FILES = [
    'data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt',
    'data/sources/lila-wam-reference-20260923/models/vla_model_fm.py',
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def js(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def parameters_sha(model):
    digest = hashlib.sha256()
    for name, param in model.state_dict().items():
        digest.update(name.encode())
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def observe_robot_roles(env):
    """Pixel-only role wrapper with explicit real Panda hand/finger membership.

    Membership does not force visibility: self-occluded fingers can have zero
    visible pixels. Geometry identities are public simulator instance masks.
    """
    obs, rgb, goal_mask = env.observe()
    actor_ids = np.asarray(env.actor_geom_ids, dtype=int)
    if len(actor_ids) == 0:
        raise ValueError('Panda hand/finger role has no bound geometry identities')
    names = [env.model.geom(i).name or '' for i in range(env.model.ngeom)]
    object_ids = [i for i, name in enumerate(names)
                  if name == 'peg' or name.startswith('cup_')]
    if not object_ids:
        raise ValueError('Manipulated object role has no bound geometry identities')
    env.renderer.enable_segmentation_rendering()
    try:
        seg = env.renderer.render().copy()
    finally:
        env.renderer.disable_segmentation_rendering()
    rendered = seg[:, :, 1] == int(env.mj.mjtObj.mjOBJ_GEOM)
    masks = np.stack([
        np.isin(seg[:, :, 0], actor_ids) & rendered,
        np.isin(seg[:, :, 0], object_ids) & rendered,
        np.asarray(goal_mask, dtype=bool),
    ])
    if rgb.shape != (448, 448, 3) or masks.shape != (3, 448, 448):
        raise ValueError('Frozen P43 visual input requires native 448px RGB/masks')
    boxes, visible = mask_geometry(masks)
    return dict(obs, role_bbox=boxes, role_visible=visible), rgb, masks.astype(np.uint8)


def validate_freeze(path, task=None):
    from prove.fast_adapter.robot_geometry_env_v05 import source_assets

    frozen = json.loads(Path(path).read_text())
    if (frozen.get('schema') != 'prove.p44-robot-pilot.v1'
            or frozen.get('status') != 'frozen'
            or frozen.get('phase') != SCOPE
            or frozen.get('root_authorized') is not True):
        raise ValueError('A root-authorized frozen P44 development protocol is required')
    if frozen['arms'] != ['fixed', 'continuous'] or frozen['model_seeds'] != [11]:
        raise ValueError('Only the declared two-arm, one-checkpoint post-repair recheck is admitted')
    if set(frozen['tasks']) != {'cup'}:
        raise ValueError('Only cup is admitted in this repair recheck')
    for task_name, cfg in frozen['tasks'].items():
        if len(cfg['seeds']) != 2 or len(set(cfg['seeds'])) != 2:
            raise ValueError('The bounded pilot admits two previously opened development conditions')
        if cfg['splits'] != ['id', 'geometry']:
            raise ValueError('The bounded pilot requires one ID and one geometry condition')
        for model_seed in frozen['model_seeds']:
            if cfg['checkpoints'][str(model_seed)] not in frozen['files']:
                raise ValueError(f'Checkpoint missing from freeze: {task_name}/{model_seed}')
        if cfg['pca'] not in frozen['files']:
            raise ValueError(f'Training-only PCA missing from freeze: {task_name}')
    for source in SOURCE:
        if source not in frozen['files']:
            raise ValueError(f'Runtime source missing from freeze: {source}')
    model_dir = Path(frozen['execution']['model_dir'])
    visual_files = VISION_FILES + [str(model_dir / name) for name in
                                   ('model.safetensors', 'config.json', 'preprocessor_config.json')]
    for filename in visual_files:
        if filename not in frozen['files']:
            raise ValueError(f'Frozen vision asset missing: {filename}')
    if not frozen.get('robot_assets'):
        raise ValueError('Robot XML/mesh asset inventory missing from freeze')
    if source_assets() != frozen['robot_assets']:
        raise ValueError('Robot XML/mesh assets changed before execution')
    for filename, digest in frozen['files'].items():
        if sha(filename) != digest:
            raise ValueError(f'Frozen component changed: {filename}')
    gate = frozen['engineering_gate']
    if sha(gate['path']) != gate['sha256']:
        raise ValueError('Engineering qualification receipt changed')
    receipt = json.loads(Path(gate['path']).read_text())
    task_gate = receipt.get('tasks', {}).get(task, {}) if task else {}
    if receipt.get('passed') is not True and task_gate.get('passed') is not True:
        raise ValueError('Robot engineering qualification has not passed')
    if frozen.get('steps_per_episode', 0) < 10:
        raise ValueError('Insufficient declared action budget')
    return frozen


def snapshot(env):
    """Record integration state in addition to human-auditable qpos/qvel."""
    spec = env.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(env.mj.mj_stateSize(env.model, spec))
    env.mj.mj_getState(env.model, env.data, state, spec)
    return dict(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                integration=state, time=float(env.data.time))


def restore_physical(env, state):
    env.mj.mj_setState(env.model, env.data, state['integration'],
                       env.mj.mjtState.mjSTATE_INTEGRATION)
    env.mj.mj_forward(env.model, env.data)


def _binding(path):
    return dict(path=str(path), sha256=sha(path))


def evaluate_sealed(env, traces, terminal):
    restore_physical(env, terminal)
    facts = env.facts()
    sustained, penetrations, all_penetrations = [], [], []
    for row in list(traces[-9:]) + [terminal]:
        restore_physical(env, row)
        sustained.append(bool(env.facts()['success']))
        all_penetrations.append(max([max(0., -float(c.dist)) for c in env.data.contact] + [0.]))
        penetrations.append(all_penetrations[-1])
    facts.update(success=bool(all(sustained) and max(penetrations) < .002),
                 last10_successes=sum(sustained), max_last10_penetration=max(penetrations),
                 max_last10_all_contact_penetration=max(all_penetrations))
    restore_physical(env, terminal)
    return facts


def rollout(task, split, seed, arm, out, model, vision, bindings, frozen, video=False):
    from prove.fast_adapter.robot_geometry_env_v05 import RobotGeometrySkill, robot_command, source_assets

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration(task, seed, split)
    steps = frozen['steps_per_episode']
    close_step = frozen.get('cup_close_step', 105)
    pre = dict(cfg=cfg, arm=arm, steps=steps, scope=SCOPE,
               sources={source: sha(source) for source in SOURCE}, bindings=bindings,
               time=time.time(), force_adaptation=False, insert_initially_firmly_held=True,
               neural_input='Visible RGB/part masks/corners, calibrated camera and own-tool pose only',
               coordinate_frame='Canonical P43 task world; no target-dependent normalization')
    js(out / 'PRE_RUN.json', pre)
    for source in SOURCE:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
    env = None
    traces, observations, frames, detail_frames = [], [], [], []
    before_hash = parameters_sha(model)
    try:
        env = RobotGeometrySkill(cfg)
        if env.model.nmocap != 0 or env.model.nu < 7:
            raise ValueError('P44 requires an actuated articulated arm with no mocap bodies')
        assets = source_assets()
        # Every robot mesh/XML actually loaded must be result-free bound.
        if frozen['robot_assets'] != assets:
            raise ValueError('Robot XML/mesh asset inventory differs from frozen qualification')
        pre['robot_assets'] = assets
        pre['actor_geom_names'] = [env.model.geom(i).name for i in env.actor_geom_ids]
        pre['robot_model'] = dict(nmocap=int(env.model.nmocap), nq=int(env.model.nq),
                                 nv=int(env.model.nv), nu=int(env.model.nu))
        js(out / 'PRE_RUN.json', pre)
        if task == 'insert':
            env.calibrate_held_geometry()
        adapter = NeuralGeometryLatent(model, arm=arm)
        anchor = None
        for t in range(steps):
            obs, rgb, masks = observe_robot_roles(env)
            if (task != 'cup' or t < close_step) and obs['corners'] is not None:
                obs['local'] = vision.encode(rgb[None], masks[None]).cpu().numpy()[0]
                adapter.update(obs, t)
            else:
                obs['local'] = np.zeros(48, np.float32)
            observations.append(obs)
            estimate = adapter.estimate()
            length = env.tool_calibration.get('length', 0.)
            if task == 'cup' and t == close_step and estimate is not None:
                p, rotation, size = estimate
                anchor = (p + rotation[:, 0] * (size[0] - .003) - rotation[:, 2] * .004,
                          rotation.copy())
            p, rotation, closed, stage = robot_command(
                task, t, estimate, env.start_pos, env.start_R, length, anchor)
            adaptation = None
            if adapter.records and adapter.records[-1]['t'] == t:
                adaptation = dict(adapter.records[-1])
                previous = pose(np.array(adaptation['pre_update_points']))
                p0, r0, _, _ = robot_command(task, t, previous, env.start_pos, env.start_R, length, anchor)
                adaptation.update(
                    same_state_action_position_delta=float(np.linalg.norm(p - p0)),
                    same_state_action_rotation_delta=float(np.linalg.norm(
                        Rotation.from_matrix(rotation @ r0.T).as_rotvec())))
            row = dict(snapshot(env), t=t, command_p=p.copy(), command_R=rotation.copy(),
                       closed=bool(closed), stage=stage, estimate=estimate, adaptation=adaptation,
                       controls_before=env.data.ctrl.copy())
            if video:
                frames.append(np.concatenate([env.render_global(), rgb], axis=1))
                detail_frames.append(env.render_detail())
            env.step(p, rotation, closed)
            row['controls_after'] = env.data.ctrl.copy()
            row['robot_tracking'] = env.robot_tracking_diagnostics()
            traces.append(row)
        terminal = snapshot(env)
        if video:
            # Outcome labels appear only over a genuine post-action terminal frame.
            _, rgb, _ = observe_robot_roles(env)
            frames.append(np.concatenate([env.render_global(), rgb], axis=1))
            detail_frames.append(env.render_detail())
        np.save(out / 'TERMINAL.npy', terminal, allow_pickle=True)
        np.save(out / 'TRACE.npy', np.array(traces, dtype=object), allow_pickle=True)
        np.save(out / 'OBS.npy', np.array(observations, dtype=object), allow_pickle=True)
        if video:
            np.savez_compressed(out / 'FRAMES.npz', frames=np.array(frames))
            np.savez_compressed(out / 'DETAIL_FRAMES.npz', frames=np.array(detail_frames))
        after_hash = parameters_sha(model)
        seal = dict(trace=sha(out / 'TRACE.npy'), observations=sha(out / 'OBS.npy'),
                    terminal=sha(out / 'TERMINAL.npy'), pre_run=sha(out / 'PRE_RUN.json'),
                    neural_parameters_before=before_hash, neural_parameters_after=after_hash)
        if video:
            seal['frames'] = sha(out / 'FRAMES.npz')
            seal['detail_frames'] = sha(out / 'DETAIL_FRAMES.npz')
        js(out / 'SEAL.json', seal)
        if before_hash != after_hash:
            raise RuntimeError('Online update changed neural weights')
        facts = evaluate_sealed(env, traces, terminal)
        changes = [r['adaptation'] for r in traces if r['adaptation'] is not None]
        result = dict(task=task, split=split, seed=seed, arm=arm, scope=SCOPE, **facts,
                      model_seed=bindings['model_seed'], steps=steps,
                      valid_frames=sum(o['corners'] is not None for o in observations),
                      actor_visible_frames=sum(bool(o['role_visible'][0]) for o in observations),
                      adaptation_updates=sum(c['U'] for c in changes),
                      max_latent_change=max([c['latent_change'] for c in changes] + [0.]),
                      max_same_state_action_position_delta=max(
                          [c['same_state_action_position_delta'] for c in changes] + [0.]),
                      neural_weights_unchanged=True, force_adaptation=False,
                      robot_tracking=env.robot_tracking_diagnostics(),
                      tool_calibration={k: v.tolist() if isinstance(v, np.ndarray) else v
                                        for k, v in env.tool_calibration.items()})
        js(out / 'RESULT.json', result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        # Preserve a failed construction/execution as engineering evidence.
        partial = {}
        if traces or observations:
            for filename, records in [('TRACE_PARTIAL.npy', traces), ('OBS_PARTIAL.npy', observations)]:
                np.save(out / filename, np.array(records, dtype=object), allow_pickle=True)
                partial[filename] = sha(out / filename)
        if env is not None:
            try:
                np.save(out / 'STATE_AT_ABORT.npy', snapshot(env), allow_pickle=True)
                partial['STATE_AT_ABORT.npy'] = sha(out / 'STATE_AT_ABORT.npy')
            except Exception:
                # An invalid simulator state must not replace the original failure.
                pass
        js(out / 'ABORT.json', dict(type=type(exc).__name__, reason=str(exc),
                                    completed_steps=len(traces), scope=SCOPE,
                                    partial_evidence_sha256=partial,
                                    interpretation='Engineering/implementation gap, not model failure'))
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--task', choices=['cup'], required=True)
    parser.add_argument('--model-seed', type=int, choices=[11], required=True)
    parser.add_argument('--seeds', help='Optional staged subset of already-frozen seeds')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    frozen = validate_freeze(args.freeze, args.task)
    cfg = frozen['tasks'][args.task]
    requested = cfg['seeds'] if args.seeds is None else [int(s) for s in args.seeds.split(',')]
    if not requested or len(requested) != len(set(requested)) or not set(requested) <= set(cfg['seeds']):
        raise ValueError('Requested conditions differ from the frozen pilot')
    torch.set_num_threads(2)
    checkpoint = cfg['checkpoints'][str(args.model_seed)]
    model, package = load_model(checkpoint, 'cpu')
    if model.task != args.task or package['seed'] != args.model_seed:
        raise ValueError('Checkpoint/task/model seed mismatch')
    vision = FrozenRoleVision(cfg['pca'], model_dir=frozen['execution']['model_dir'],
                              device=args.device, microbatch=1)
    bindings = dict(checkpoint=_binding(checkpoint), pca=_binding(cfg['pca']),
                    freeze=_binding(args.freeze), engineering_gate=frozen['engineering_gate'],
                    model_seed=args.model_seed)
    output = Path(frozen['execution']['out']) / f'{args.task}_seed{args.model_seed}'
    for seed, split in zip(cfg['seeds'], cfg['splits']):
        if seed not in requested:
            continue
        pair = dict(task=args.task, model_seed=args.model_seed, seed=seed, split=split)
        video = pair in frozen['media_pairs']
        for arm in frozen['arms']:
            rollout(args.task, split, seed, arm, output / f'{split}_{seed}_{arm}',
                    model, vision, bindings, frozen, video=video)


if __name__ == '__main__':
    main()
