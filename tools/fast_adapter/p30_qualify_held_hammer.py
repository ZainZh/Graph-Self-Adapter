"""Bounded, privileged environment qualification; NOT an adapter evaluation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from prove.fast_adapter.object_interaction_env import object_config
from prove.fast_adapter.held_hammer_env import HeldHammerEnv, PreparationFailed

ROOT = Path('evaluations/fast_adapter_p30_online_v01')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run():
    scope = json.loads((ROOT/'EXPLORATION_SCOPE.json').read_text())
    cfg = scope['environment_qualification']
    out = ROOT/'held_initialization_v1'
    out.mkdir(parents=True, exist_ok=False)
    paths = [ROOT/'EXPLORATION_SCOPE.json', Path(__file__),
             Path('prove/fast_adapter/held_hammer_env.py'),
             Path('prove/fast_adapter/object_interaction_env.py')]
    save(out/'SOURCE_FREEZE.json', dict(time=time.time(), result_unopened=True,
         bindings=[dict(path=str(p), sha256=digest(p)) for p in paths]))
    rows = []
    for si, split in enumerate(cfg['strata']):
        for i in range(cfg['conditions_per_stratum']):
            geom = object_config('hammer-block', cfg['object_seed_origin'] + si*100 + i, split)
            seed = cfg['reset_seed_origin'] + si*100 + i
            env = HeldHammerEnv(geom, render=i == 0,
                    preparation_budget=cfg['preparation_max_control_steps'],
                    required_streak=cfg['required_bilateral_streak'],
                    required_lift=cfg['required_lift_m'])
            prefix = f'{split}_{i:02d}'
            row = dict(split=split, index=i, config=geom, reset_seed=seed)
            try:
                env.reset(seed)
            except PreparationFailed:
                row.update(preparation_passed=False, interaction_success=None)
            else:
                row['preparation_passed'] = True
                row['initial_contacts'] = list(map(bool, env.env.contacts()))
                if i == 0:
                    Image.fromarray(env.rgb()).save(out/f'{prefix}_held.png')
                states = [env.state()]; actions = []
                for _ in range(cfg['execution_check_max_control_steps']):
                    env.step(env.teacher())
                    states.append(env.state()); actions.append(env.last_executed_action.copy())
                # Trajectory is stored before outcome aggregation; GT teacher is
                # explicit, not passed off as a learned policy qualification.
                np.savez_compressed(out/f'{prefix}_interaction.npz', actions=actions,
                    **{k: np.stack([s[k] for s in states]) for k in states[0]})
                row['interaction_success'] = any(m['success'] for m in env.metrics)
                row['task_steps'] = len(actions)
                row['metrics'] = env.metrics
                if i == 0:
                    Image.fromarray(env.rgb()).save(out/f'{prefix}_after.png')
            prep = env.preparation
            np.savez_compressed(out/f'{prefix}_preparation.npz', actions=prep['actions'],
                **{k: np.stack([s[k] for s in prep['states']]) for k in prep['states'][0]})
            row['preparation_steps'] = prep['steps']
            row['preparation_metrics'] = prep['metrics']
            save(out/f'{prefix}.json', row)
            rows.append(row)
            print(prefix, row['preparation_passed'], row['interaction_success'],
                  'preparation_steps', prep['steps'], flush=True)
            env.close()
    summary = dict(phase='privileged_environment_qualification', n=len(rows),
       preparation_passed=sum(r['preparation_passed'] for r in rows),
       teacher_interaction_success=sum(r['interaction_success'] is True for r in rows),
       learned_policy_tested=False, adapter_tested=False,
       excluded_conditions=0, rows=[{k: v for k, v in r.items() if 'metrics' not in k} for r in rows])
    save(out/'RESULT.json', summary)
    print('QUALIFICATION_COMPLETE', {k: v for k, v in summary.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    run()
