"""Task-scoped output path for the frozen P29 auditor and explicit active candidate."""
import argparse
import json
from pathlib import Path
from tools.fast_adapter.p29_common import task_root


def main(task, phase, rnd):
    if task == 'hammer-block' and rnd == 'r2':
        from tools.fast_adapter.p29_hammer_candidate_r2 import configure
        configure()
    elif task == 'hammer-block':
        from tools.fast_adapter.p29_candidate_v04 import configure
        configure(task_root(task)/'candidates/base_v03/CANDIDATE_V2.json')
    elif rnd == 'r2':
        from tools.fast_adapter.p29_candidate_r2 import configure
        configure()
    from tools.fast_adapter import p29_audit
    old = p29_audit.save_json
    def save(path, value):
        path = Path(path)
        if path.name == 'AUDIT.json':
            path = path.parent/task/'AUDIT.json'
            assert not path.exists(), path
        return old(path, value)
    p29_audit.save_json = save
    p29_audit.audit(phase, rnd, [task])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--task', required=True)
    p.add_argument('--phase', default='final'); p.add_argument('--round', default='r1'); a = p.parse_args()
    main(a.task, a.phase, a.round)
