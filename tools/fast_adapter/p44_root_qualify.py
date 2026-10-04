"""Independent replay and visible-input checks for the completed P44 engineering cells."""
import hashlib
import json
from pathlib import Path
import numpy as np

from prove.fast_adapter.robot_geometry_env import RobotGeometrySkill
from tools.fast_adapter.p44_evaluate import observe_robot_roles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('evaluations/fast_adapter_p44_robot_geometry_v01')
    rows = []
    for task, seed, split in [('cup', 440000, 'id'), ('cup', 440100, 'geometry'),
                             ('insert', 441000, 'id'), ('insert', 441100, 'geometry')]:
        cell = root / 'engineering' / f'render_v02_{task}_{split}{seed}'
        pre = json.loads((cell / 'PRE_RUN.json').read_text())
        seal = json.loads((cell / 'SEAL.json').read_text())
        result = json.loads((cell / 'RESULT.json').read_text())
        for name, digest in seal.items():
            assert sha(cell / name) == digest, name
        for name, digest in pre['sources'].items():
            assert sha(name) == digest and sha(cell / 'source' / name) == digest, name
        trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
        terminal = np.load(cell / 'TERMINAL.npy', allow_pickle=True).item()
        env = RobotGeometrySkill(pre['config'], render=True)
        maximum = {'qpos': 0., 'qvel': 0.}
        try:
            obs, rgb, masks = observe_robot_roles(env)
            public_input = {'first_corners_valid': obs['corners'] is not None,
                            'role_pixels': masks.sum(axis=(1, 2)).tolist()}
            for row in trace:
                for key in maximum:
                    maximum[key] = max(maximum[key], float(np.max(np.abs(getattr(env.data, key) - row['state'][key]))))
                env.step(row['command_p'], row['command_R'], row['closed'])
            for key in maximum:
                maximum[key] = max(maximum[key], float(np.max(np.abs(getattr(env.data, key) - terminal[key]))))
            successes, penetrations = [], []
            for state in [r['state'] for r in trace[-9:]] + [terminal]:
                env.restore(state)
                successes.append(env.facts()['success'])
                penetrations.append(max([max(0., -float(c.dist)) for c in env.data.contact] + [0.]))
            recomputed = bool(all(successes) and max(penetrations) < .002)
            matched = recomputed == result['success'] and sum(successes) == result['last10_successes']
            prefix_visible = sum(r.get('visible_corners') is not None for r in trace[:12])
            passed = bool(matched and recomputed and max(maximum.values()) < 1e-6
                          and env.model.nmocap == 0 and env.model.nu == 9
                          and result['max_arm_joint_limit_violation_rad'] == 0
                          and result['max_finger_joint_limit_violation_m'] == 0
                          and prefix_visible >= 3)
            # Calibration invokes mj_forward. Keep this extra diagnostic out of
            # the original Oracle replay's integration sequence.
            env.restore(trace[0]['state'])
            if task == 'insert':
                calibration = env.calibrate_held_geometry()
                public_input['held_length_from_mask'] = float(calibration['length'])
                public_input['calibration_pixel_residual'] = float(calibration['pixel_residual'])
                # Truth is diagnosis only; no fit or command receives this error.
                public_input['diagnostic_length_error_m'] = float(calibration['length'] - pre['config']['length'])
            row = dict(task=task, seed=seed, split=split, cell=str(cell), passed=passed,
                       actual_seven_joint_actuation=True, physical_replay_maximum_error=maximum,
                       success_recomputed=recomputed, evaluator_matches=matched,
                       initial12_valid_corner_frames=prefix_visible, public_input=public_input,
                       result_sha256=sha(cell / 'RESULT.json'), pre_run_sha256=sha(cell / 'PRE_RUN.json'))
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    gate = dict(schema='prove.p44-root-engineering-gate.v1',
                passed=all(row['passed'] for row in rows),
                tasks={task: {'passed': all(row['passed'] for row in rows if row['task'] == task)}
                       for task in ('cup', 'insert')},
                conditions=rows, neural_outcomes_opened=False,
                scope='Oracle engineering only; articulated execution and usable public inputs, not neural benefit',
                source_sha256=sha(__file__))
    output = root / 'ROOT_ENGINEERING_GATE_V02.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + '\n')
    if not gate['passed']:
        raise SystemExit('Engineering qualification failed')


if __name__ == '__main__':
    main()
