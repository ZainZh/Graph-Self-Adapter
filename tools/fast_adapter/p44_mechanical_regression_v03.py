"""P44 v03 empty-close and opened-command mechanical regression only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from prove.fast_adapter.robot_geometry_env_v03 import RobotGeometrySkill, configuration, source_assets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def run(out, source_cell=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    originals = {}
    if source_cell:
        cell = Path(source_cell)
        cfg = json.loads((cell / 'PRE_RUN.json').read_text())['cfg']
        old = np.load(cell / 'TRACE.npy', allow_pickle=True)
        mode = 'counterfactual_replay_of_opened_neural_commands'
        originals = {str(cell / n): sha(cell / n) for n in ('PRE_RUN.json', 'TRACE.npy', 'TERMINAL.npy', 'SEAL.json')}
        steps = len(old)
    else:
        cfg = configuration('cup', 440000, 'id')
        old = None
        mode = 'empty_close_at_nominal_observation_pose'
        steps = 30
    sources = ['prove/fast_adapter/robot_geometry_env_v03.py', 'prove/fast_adapter/geometry3d_env.py', 'tools/fast_adapter/p44_mechanical_regression_v03.py']
    js(out / 'PRE_RUN.json', dict(scope='engineering_v03_mechanical_repair_regression', mode=mode, config=cfg, steps=steps, timestamp=time.time(), sources={p: sha(p) for p in sources}, robot_assets=source_assets(), original_sealed_files=originals, limit_tolerance_m=.0001, fresh_neural_evidence=False, change='complete the four opposing-finger body-pair exclusions only'))
    for source in sources:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
    env = RobotGeometrySkill(cfg, render=False)
    trace = []
    try:
        for i in range(steps):
            p, R, closed = (env.start_pos, env.start_R, True) if old is None else (old[i]['command_p'], old[i]['command_R'], old[i]['closed'])
            state = env.snapshot()
            trace.append(dict(state=state, command_p=p.copy(), command_R=R.copy(), closed=bool(closed)))
            env.step(p, R, closed)
        terminal = env.snapshot()
        np.save(out / 'TRACE.npy', np.array(trace, dtype=object), allow_pickle=True)
        np.save(out / 'TERMINAL.npy', terminal, allow_pickle=True)
        js(out / 'SEAL.json', {name: sha(out / name) for name in ('TRACE.npy', 'TERMINAL.npy')})
        finger_ids = {env.model.geom(n).id for n in ['finger1_collision', 'finger2_collision', 'finger1_pad_collision', 'finger2_pad_collision']}
        object_ids = set(np.nonzero(env.model.geom_bodyid == env.obj_id)[0].tolist())
        violations = []; arm_violations = []; object_contact_states = []; internal_contact_states = []; penetrations = []; finger_q = []
        for state in [row['state'] for row in trace] + [terminal]:
            env.restore(state)
            diagnostics = env.robot_tracking_diagnostics()
            violations.append(diagnostics['finger_joint_limit_violation_m'])
            arm_violations.append(diagnostics['arm_joint_limit_violation_rad'])
            finger_q.append(env.data.qpos[env.finger_q].copy())
            object_contact_states.append(any((c.geom1 in finger_ids and c.geom2 in object_ids) or (c.geom2 in finger_ids and c.geom1 in object_ids) for c in env.data.contact))
            internal_contact_states.append(any(c.geom1 in finger_ids and c.geom2 in finger_ids for c in env.data.contact))
            penetrations.append(max([max(0., -float(c.dist)) for c in env.data.contact] + [0.]))
        q = np.array(finger_q)
        result = dict(scope='engineering_v03_mechanical_repair_regression', mode=mode, original_source_cell=source_cell, fresh_neural_evidence=False, max_finger_joint_limit_violation_m=max(violations), max_arm_joint_limit_violation_rad=max(arm_violations), finger_q_min=q.min(axis=0).tolist(), finger_q_max=q.max(axis=0).tolist(), final_finger_q=q[-1].tolist(), object_contact_states=sum(object_contact_states), internal_finger_contact_states=sum(internal_contact_states), max_all_contact_penetration=max(penetrations), terminal_task_facts=env.facts(), mechanical_check_passed=bool(max(violations) < .0001 and max(arm_violations) < .001 and not any(internal_contact_states)))
        if old is None:
            result['empty_close_passed'] = bool(result['mechanical_check_passed'] and not any(object_contact_states) and max(abs(q[-1])) < .001)
        else:
            original_terminal = np.load(Path(source_cell) / 'TERMINAL.npy', allow_pickle=True).item()
            result['original_terminal_finger_q'] = original_terminal['qpos'][env.finger_q].tolist()
            result['not_an_online_neural_rerun'] = True
        js(out / 'RESULT.json', result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--source-cell')
    args = p.parse_args()
    run(args.out, args.source_cell)
