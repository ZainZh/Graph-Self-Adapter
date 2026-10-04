"""Engineering-only qualification; never included in P38 scientific denominators."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from prove.fast_adapter.near_contact_push import NearPush, configuration, nominal_force

ROOT = Path('evaluations/fast_adapter_p38_causal_v01/push_engineering')


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    results = []
    for split in ('native', 'train', 'ood'):
        for seed in range(4):
            cfg = configuration(38000+seed, split)
            for mode in ('base', 'oracle'):
                env = NearPush(cfg)
                trace, frames = [], []
                for t in range(30):
                    obs, own, rgb = env.public()
                    x, goal = obs['positions'][1:]
                    force = nominal_force(goal-x) if mode == 'base' else nominal_force(
                        goal-x, cfg['damping']+.2, cfg['friction']+.01)
                    frames.append(rgb) if seed == 0 and t in (0, 10, 29) else None
                    # Truth retained for post-runtime engineering diagnostics only.
                    trace.append({'t': t, 'rgb_positions': obs['positions'].tolist(),
                                  'force': force, 'state': {k: v.tolist() if hasattr(v,'tolist') else v for k,v in env.state().items()}})
                    env.step(force)
                end = env.state(); error = abs(end['object_x']-env.goal)
                row = {'split': split, 'seed': seed, 'mode': mode, 'config': cfg,
                       'final_error': error, 'speed': float(end['qvel'][1]),
                       'success': error < .0015 and abs(end['qvel'][1]) < .005,
                       'trace': trace}
                results.append(row); print({k:v for k,v in row.items() if k not in ('trace','config')}, flush=True)
                if frames:
                    Image.fromarray(np.concatenate(frames, axis=1)).save(ROOT/f'{split}_{mode}.jpg')
                env.close()
    (ROOT/'RESULT.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
