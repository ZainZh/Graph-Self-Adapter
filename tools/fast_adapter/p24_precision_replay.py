"""Resolve cached-fp16 audit exceptions by deterministic sealed-action replay.

Post-result evidence reconstruction only: no new policy trajectory, model
selection, training or success labels. Original numerical threshold stays fixed.
Use the original server/GPU and encoder batch size; save recovered float32 inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import geometry
from tools.fast_adapter.p24_common import R, load_base, load_adapter
from tools.fast_adapter.p24_rollout import SceneBatch, visual_assets
from tools.fast_adapter.p18_run import digest, save_json


def main(dino):
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    audit = json.loads((R/'summary/AUDIT_SUPPLEMENT.json').read_text())
    groups = {}
    for item in audit['precision_exceptions']:
        groups.setdefault((item['root'], item['arm']), set()).update(
            index[1] for index in item['indices'])
    encoder, pca = visual_assets(dino)
    base = load_base()
    rows = []
    for (root_s, arm), steps in groups.items():
        root = Path(root_s)
        rawpath = root/f'{arm}_RAW.pt'
        raw = torch.load(rawpath, weights_only=False)
        trace = torch.load(root/'SUPPORT_SEALED.pt', weights_only=True)
        n, ss, st = len(raw['positions']), raw['seed_base'], raw['stratum']
        seed = int(root.name[4:])
        net = load_adapter(seed, 'history' if arm.startswith('history') else 'meta')
        z = trace['latents'][arm].cuda()
        scene = SceneBatch([geometry(ss+i, st) for i in range(n)],
                           np.arange(n)+ss+100000, encoder, pca)
        scene.reset(np.arange(n)+ss+300000)
        initial_rgb = np.stack([e.rgb() for e in scene.envs[:2]])
        initial_rgb_error = int(np.max(np.abs(initial_rgb.astype(int)-raw['audit_rgb'][0].astype(int))))
        state_error, velocity_error = 0., 0.
        for t in range(max(steps)+1):
            positions = np.stack([e.state() for e in scene.envs])
            velocities = np.stack([e.data.qvel.copy() for e in scene.envs])
            state_error = max(state_error, float(np.max(np.abs(positions-raw['positions'][:, t]))))
            velocity_error = max(velocity_error, float(np.max(np.abs(velocities-raw['velocities'][:, t]))))
            if t in steps:
                v, p = scene.observe()
                cached = raw['visual'][:, t].cuda().float()
                previous = torch.tensor(raw['actions'][:, t-1], device='cuda') if t else torch.zeros_like(p)
                with torch.no_grad():
                    ba = base(v, p)
                    predicted = net.action(v, p, previous, ba, z)
                    # Exactly reproduce the original audit calculation, using cached base.
                    half_predicted = net.action(cached, p, previous, raw['base'][:, t].cuda(), z)
                saved = torch.tensor(raw['actions'][:, t], device='cuda')
                error = float((predicted-saved).abs().max())
                base_error = float((ba-raw['base'][:, t].cuda()).abs().max())
                cache_equal = torch.equal(v.cpu().half(), raw['visual'][:, t])
                packet = R/'summary'/f'PRECISION_REPLAY_{st}_seed{seed}_{arm}_t{t}.pt'
                torch.save({'visual_float32': v.cpu(), 'proprio': p.cpu(), 'previous_action': previous.cpu(),
                    'z': z.cpu(), 'base': ba.cpu(), 'action': predicted.cpu(),
                    'action_from_cached_half_visual': half_predicted.cpu(),
                    'original_action': saved.cpu(), 'rgb': torch.tensor(scene.last_rgb),
                    'source_raw': str(rawpath), 'source_raw_sha256': digest(rawpath)}, packet)
                rows.append({'root': root_s, 'arm': arm, 'time_step': t, 'batch_size': n,
                    'replayed_state_max_error': state_error, 'replayed_velocity_max_error': velocity_error,
                    'initial_audit_rgb_max_error': initial_rgb_error,
                    'recovered_float32_cast_to_saved_half_bitwise_equal': cache_equal,
                    'maximum_base_error': base_error, 'maximum_action_error': error,
                    'half_visual_action_max_error': float((half_predicted-saved).abs().max()),
                    'float32_to_half_feature_max_error': float((v-cached).abs().max()),
                    'packet': str(packet), 'packet_sha256': digest(packet),
                    'passed': bool(cache_equal and state_error < 1e-10 and velocity_error < 1e-10
                                   and initial_rgb_error == 0 and error < .001 and base_error < .001)})
                print(json.dumps(rows[-1]), flush=True)
            if t < max(steps):
                # Replay recorded commands only; reconstructed predictions never drive physics.
                scene.step(raw['actions'][:, t])
        scene.close()
    report = {'passed': bool(rows) and all(r['passed'] for r in rows), 'records': rows,
        'original_threshold_changed': False, 'model_or_result_changed': False,
        'scope': 'Post-result deterministic replay of sealed actions and original encoder; no new evaluation episodes.',
        'source': {'path': __file__, 'sha256': digest(Path(__file__))},
        'supplement_sha256': digest(R/'summary/AUDIT_SUPPLEMENT.json')}
    save_json(R/'summary/PRECISION_REPLAY.json', report)
    assert report['passed'], report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dino', default='assets/dinov3')
    main(parser.parse_args().dino)
