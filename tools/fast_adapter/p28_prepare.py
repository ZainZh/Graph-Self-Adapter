"""Bounded task preparation: official teacher, demonstrations, frozen RGB base."""
import argparse
import os
import subprocess
import sys
import time
from tools.fast_adapter.p28_common import R, TASKS, task_root, save_json, digest


def run(task, module, *args):
    command = [sys.executable, '-m', module, '--task', task, *args]
    subprocess.run(command, check=True)


def main(task):
    began = time.time()
    run(task, 'tools.fast_adapter.p28_data', 'qualify')
    run(task, 'tools.fast_adapter.p28_data', 'collect', '--stage', 'demos')
    run(task, 'tools.fast_adapter.p28_data', 'merge')
    run(task, 'tools.fast_adapter.p28_train', 'base', '--stage', 'bc')
    run(task, 'tools.fast_adapter.p28_data', 'collect', '--stage', 'dagger')
    run(task, 'tools.fast_adapter.p28_train', 'base', '--stage', 'dagger')
    root = task_root(task)
    save_json(root/'BASE_READY.json', {'task': task, 'seconds': time.time()-began,
              'base_sha256': digest(root/'models/base_final.pt'),
              'pca_sha256': digest(root/'data/PCA.pt'),
              'status': 'trained_pending_native_rollout_qualification'})
    print('BASE_PREPARATION_COMPLETE', task, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--task', required=True, choices=TASKS)
    main(p.parse_args().task)
