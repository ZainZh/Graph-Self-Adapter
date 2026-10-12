#!/usr/bin/env python3
"""Shared cup/apple/cube frontdoor; real is currently offline contract checking."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (REPO, HERE):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

TASKS = ('put_mug_on_box', 'collect_apple', 'collect_cubes')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--backend', choices=('sim', 'real'), default='sim')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--real-config', type=Path, default=REPO/'evaluations/gemini_graph_v19/CONFIG_PROPOSED.json')
    args, remainder = parser.parse_known_args(argv)
    if args.backend == 'real':
        if not args.check: parser.error('real execution is not currently authorized; use --check for offline constructors/schema tests')
        if remainder: parser.error('real currently accepts offline contract checking only')
        from three_task_v25_public_io import offline_contract_check, real_constructor_check
        result = offline_contract_check(args.task)
        config=json.loads(args.real_config.read_text())
        result['real_constructor_check']=real_constructor_check(config.get('runtime',config))
        print(json.dumps(result, indent=2)); return result
    if args.task == 'put_mug_on_box':
        from cup_v25_real_capability_one_command import main as run
        return run(['sim'] + (['--check'] if args.check else []) + remainder)
    if args.task=='collect_cubes':
        from ordinary_v25_repair_entry import main as run
    else:
        from ordinary_v25_entry import main as run
    return run(['--task', args.task, 'check' if args.check else 'sim'] + remainder)

if __name__ == '__main__': main()
