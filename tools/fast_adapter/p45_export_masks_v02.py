"""Public-pixel export from sealed engineering states; no action execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.fast_adapter.p45_common import make_env, sha, js


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cell', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); cell = Path(a.cell); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    pre = json.loads((cell / 'PRE_RUN.json').read_text()); cfg = pre.get('config', pre.get('cfg'))
    seal = json.loads((cell / 'SEAL.json').read_text())
    assert sha(cell / 'TRACE.npy') == seal.get('TRACE.npy', seal.get('trace'))
    trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
    import mujoco
    original = mujoco.mj_step; calls = []
    mujoco.mj_step = lambda *args, **kwargs: calls.append('suppressed constructor integration')
    try:
        env = make_env(cfg['task'], cfg['seed'], cfg['split'])
    finally:
        mujoco.mj_step = original
    images, masks, parts, obsrows = [], [], [], []
    try:
        for row in trace[:12]:
            env.restore(row['state'])
            obs, rgb, mask = env.observe()
            env.renderer.enable_segmentation_rendering(); seg = env.renderer.render().copy()
            env.renderer.disable_segmentation_rendering()
            ids = list(env.goal_ids)
            visible = seg[:, :, 1] == int(env.mj.mjtObj.mjOBJ_GEOM)
            parts.append(np.stack([(seg[:, :, 0] == gid) & visible for gid in ids]))
            images.append(rgb); masks.append(mask); obsrows.append(obs)
        np.savez_compressed(out / 'PUBLIC_PIXELS.npz', rgb=np.asarray(images), mask=np.asarray(masks), parts=np.asarray(parts),
                            **{k: np.asarray([o[k] for o in obsrows]) for k in ('camera_position', 'camera_rotation', 'K')})
        js(out / 'MANIFEST.json', dict(source=str(cell), source_trace_sha256=sha(cell / 'TRACE.npy'),
            output_sha256=sha(out / 'PUBLIC_PIXELS.npz'), frames=len(images), suppressed_constructor_steps=len(calls),
            physics_steps_executed=0, no_target_truth_in_pixels=True, generator_sha256=sha(__file__)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
