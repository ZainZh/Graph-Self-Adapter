"""Evaluate every preregistered step count; no training or outcome-based selection."""
from pathlib import Path
import json
import time
from tools.fast_adapter.p18_run import save_json, digest
from tools.fast_adapter import p22_batch

R = Path('evaluations/fast_adapter_p23_update_steps_v01')


def main():
    p22_batch.R = R
    p22_batch.verify_sources()
    protocol = json.loads((R / 'PROTOCOL.json').read_text())
    began = time.time()
    paths = [R / 'SOURCE_FREEZE.json', R / 'PROTOCOL.json',
             R / 'bank/PCA.pt', R / 'models/base/model.pt']
    paths += [R / 'models' / f'seed{s}' / f'{arm}.pt'
              for s in protocol['model_seeds'] for arm in ('history', 'hybrid')]
    assert not (R / 'FINAL_FREEZE.json').exists()
    save_json(R / 'FINAL_FREEZE.json', {
        'time': time.time(), 'outcomes_opened': False, 'training_updates': 0,
        'bindings': [{'path': str(p), 'sha256': digest(p)} for p in paths]})
    for seed in [0] + protocol['model_seeds']:
        p22_batch.parallel([
            (gpu, f'final_{st}_seed{seed}', [
                'tools.fast_adapter.p23_eval', '--stage', 'final',
                '--stratum', st, '--seed', str(seed), '--dino', 'assets/dinov3'])
            for gpu, st in enumerate(protocol['final']['strata'])])
    p22_batch.verify_sources()
    save_json(R / 'CAMPAIGN_DONE.json', {
        'status': 'completed_pending_root_audit', 'final_run': True,
        'seconds': time.time() - began, 'training_updates': 0,
        'all_prespecified_steps_run': protocol['update_steps']})
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
