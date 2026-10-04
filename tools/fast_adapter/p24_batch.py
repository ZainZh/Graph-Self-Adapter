"""Finite three-GPU training and all-arm independent evaluation."""
from pathlib import Path
import json
import time
from tools.fast_adapter.p24_common import R, PARENT, SEEDS
from tools.fast_adapter.p18_run import save_json, digest
from tools.fast_adapter import p22_batch


def main():
    p22_batch.R = R; p22_batch.verify_sources(); began = time.time()
    p22_batch.parallel([(gpu, f'train_seed{seed}', ['tools.fast_adapter.p24_train', '--seed', str(seed)])
                        for gpu, seed in enumerate(SEEDS)])
    p22_batch.verify_sources()
    paths = [R/'PROTOCOL.json', R/'SOURCE_FREEZE.json', PARENT/'bank/PCA.pt', PARENT/'models/base/model.pt']
    paths += [R/'models'/f'seed{s}'/f'{name}.pt' for s in SEEDS for name in ('meta', 'history')]
    assert not (R/'FINAL_FREEZE.json').exists()
    save_json(R/'FINAL_FREEZE.json', {'time': time.time(), 'outcomes_opened': False,
        'checkpoint_selection': 'Exactly 50000 updates, all seeds, no final-based selection',
        'bindings': [{'path': str(p), 'sha256': digest(p)} for p in paths]})
    for seed in (0, *SEEDS):
        p22_batch.parallel([(gpu, f'final_{st}_seed{seed}', ['tools.fast_adapter.p24_rollout', 'evaluate',
                             '--stratum', st, '--seed', str(seed), '--dino', 'assets/dinov3'])
                            for gpu, st in enumerate(('id', 'position', 'scale'))])
    p22_batch.verify_sources()
    save_json(R/'CAMPAIGN_DONE.json', {'status': 'completed_pending_root_audit',
        'final_run': True, 'seconds': time.time()-began})
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
