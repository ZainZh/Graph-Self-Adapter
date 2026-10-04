"""Small RGB-to-runtime-feature replay from already collected P43 data."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from prove.fast_adapter.neural_geometry_visual import FrozenRoleVision
from tools.fast_adapter.p43_data import ROOT, sha, js


def main():
    start = time.monotonic()
    vision = FrozenRoleVision(microbatch=8)
    records, examples = [], []
    for task in ('cup', 'insert'):
        pca_path = ROOT/'data'/task/'role_pca.pt'
        vision.set_pca(torch.load(pca_path, map_location='cpu', weights_only=True))
        for split in ('train', 'dev'):
            pack = np.load(ROOT/'data'/task/split/'observations.npz', allow_pickle=False)
            for episode in (0, len(pack['seeds'])-1):
                seed = int(pack['seeds'][episode])
                archive = ROOT/'data_raw'/task/split/str(seed)/'visible_images.npz'
                with np.load(archive, allow_pickle=False) as raw:
                    indices = [0, len(raw['rgb'])-1]
                    # Reproduce the original batch8 contexts exactly. Batch2
                    # BF16 sensitivity is preserved separately, not hidden by
                    # enlarging the feature-replay tolerance.
                    context = np.r_[np.arange(8), np.arange(40,48)]
                    rgb_context = raw['rgb'][context]
                    mask_context = np.unpackbits(raw['role_masks_packed'][context], axis=-1)
                    rgb = rgb_context[[0,15]]
                features = vision.encode(rgb_context, mask_context).cpu().numpy()[[0,15]]
                expected = pack['local'][episode, indices]
                error = float(np.abs(features-expected).max())
                passed = bool(np.allclose(features, expected, rtol=1e-4, atol=2e-4))
                records.append(dict(task=task, split=split, seed=seed, frame_indices=indices,
                    times=pack['t'][episode, indices].tolist(), feature_absmax=error, passed=passed,
                    image_archive_sha256=sha(archive), pca_sha256=sha(pca_path)))
                if episode == 0:
                    for j, index in enumerate(indices):
                        image = Image.fromarray(rgb[j]); draw = ImageDraw.Draw(image)
                        corners = pack['corners'][episode, index]
                        if pack['corners_valid'][episode, index]:
                            draw.line([tuple(x) for x in np.r_[corners, corners[:1]]], fill=(255, 80, 60), width=2)
                        examples.append((f'{task} {split} seed{seed} t{pack["t"][episode,index]}', image))
    canvas = Image.new('RGB', (4*448, 2*480), (25, 30, 38))
    draw = ImageDraw.Draw(canvas)
    for i, (label, example) in enumerate(examples):
        x = (i%4)*448; y = (i//4)*480
        draw.text((x+8,y+8), label, fill=(235,235,235));canvas.paste(example, (x,y+32))
    canvas.save(ROOT/'visual/DATA_EXAMPLES.jpg', quality=92)
    result = dict(passed=all(x['passed'] for x in records), frames=sum(len(x['frame_indices']) for x in records),
                  original_batch8_context_preserved=True,
                  full_frozen_rgb_lila_pca_recomputed=True, test_frames_used=False,
                  maximum_feature_abs_difference=max(x['feature_absmax'] for x in records),
                  elapsed_seconds=time.monotonic()-start, records=records)
    js(ROOT/'visual/FEATURE_REPLAY.json', result)
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise RuntimeError('RGB feature replay mismatch')


if __name__ == '__main__':
    main()
