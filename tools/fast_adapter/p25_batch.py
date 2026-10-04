"""Finite P25 small development campaign; no final or automatic expansion."""
import json
import time
from pathlib import Path
from tools.fast_adapter import p22_batch
from tools.fast_adapter.p25_common import R, SEEDS, VARIANTS
from tools.fast_adapter.p18_run import digest, save_json


def main():
    p22_batch.R = R; p22_batch.verify_sources(); began = time.time()
    p22_batch.parallel([(gpu, f'train_seed{seed}', ['tools.fast_adapter.p25_train', '--seed', str(seed)]) for gpu,seed in enumerate(SEEDS)])
    p22_batch.verify_sources()
    paths = [R/'PROTOCOL.json', R/'SOURCE_FREEZE.json']
    paths += [R/'models'/f'seed{seed}'/f'{name}.pt' for seed in SEEDS for name in VARIANTS]
    save_json(R/'MODEL_FREEZE.json', {'time': time.time(), 'development_not_opened': True,
        'bindings': [{'path': str(p), 'sha256': digest(p)} for p in paths]})
    p22_batch.parallel([(gpu, f'dev_{st}', ['tools.fast_adapter.p25_eval', '--stratum', st]) for gpu,st in enumerate(('id','position','scale'))])
    p22_batch.verify_sources()
    save_json(R/'CAMPAIGN_DONE.json', {'status': 'small_development_complete_pending_review', 'seconds': time.time()-began,
        'final_started': False, 'queries': 1116, 'support_trajectories': 36})
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__': main()
