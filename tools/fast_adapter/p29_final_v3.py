"""Select a completed development candidate and seal the unchanged final protocol.
No evaluation or result-dependent sample allocation is performed by this module.
"""
import argparse
import json
import time
from pathlib import Path

from tools.fast_adapter.p29_common import R, TASKS, SEEDS, digest, save_json, task_root
from tools.fast_adapter.p29_eval import STRATA
from tools.fast_adapter.p29_report import arrays


def freeze_task(task, rounds):
    root = task_root(task)
    assert not (root/'FINAL_FREEZE.json').exists()
    assert not (root/'final').exists()
    candidates = []
    files = [R/'PROTOCOL.json', R/'EVALUATION_SOURCE_FREEZE.json',
             R/'ANALYSIS_FREEZE.json', R/'SEED_NAMESPACE_AMENDMENT.json',
             Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__)]
    for order, rnd in enumerate(rounds):
        by = {}
        for st in STRATA:
            p = root/'development'/rnd/st/'RESULT.json'
            by[st] = arrays(json.loads(p.read_text()))
            files.append(p)
        candidates.append({'round': rnd, 'geometry_ood_correct_rate': float(by['geometry_ood']['fast_correct'].mean()),
                           'canonical_harms': int(((by['canonical']['base'][None] > 0) & (by['canonical']['fast_correct'] == 0)).sum()),
                           'order': order})
    chosen = max(candidates, key=lambda d: (d['geometry_ood_correct_rate'], -d['canonical_harms'], -d['order']))
    rnd = chosen['round']
    gate = root/'candidates/base_v03/BASE_GATE.json' if task == 'hammer-block' else root/'qualification/BASE.json'
    g = json.loads(gate.read_text())
    assert g['successes'] >= 6 and g['n'] == 8
    dev_freeze = root/f'DEV_{rnd}_FREEZE.json'
    prior_bindings = json.loads(dev_freeze.read_text())['bindings']
    files += [gate, dev_freeze, root/'data/PCA.pt']
    files += [root/'models'/rnd/f'seed{s}'/f'{v}.pt' for s in SEEDS for v in ('fast', 'history', 'robot')]
    candidate = root/'candidates/base_v03/CANDIDATE_V2.json' if task == 'hammer-block' else None
    if candidate:
        cfg = json.loads(candidate.read_text()); files += [candidate, Path(cfg['base_checkpoint'])]
    else:
        files.append(root/'models/base_final.pt')
    bindings = {b['path']: b for b in prior_bindings}
    for p in files:
        b = {'path': str(p), 'sha256': digest(p)}
        if b['path'] in bindings:
            assert b == bindings[b['path']]
        bindings[b['path']] = b
    for b in bindings.values():
        assert digest(b['path']) == b['sha256'], b['path']
    ti = TASKS.index(task)
    strata = {}
    for si, st in enumerate(STRATA):
        n = 32 if st == 'canonical' else 64
        origin = 29910000 + ti*100000 + si*10000
        strata[st] = {name: list(range(origin + delta, origin + delta + n)) for name, delta in
                      [('object', 0), ('donor_object', 1000000), ('support_reset', 2000000),
                       ('donor_support_reset', 3000000), ('query_reset', 4000000)]}
    command = ['tools.fast_adapter.p29_eval', '--task', task, '--phase', 'final', '--round', rnd]
    if candidate and rnd == 'r3':
        command = ['tools.fast_adapter.p29_hammer_candidate_r3', *command]
    elif candidate and rnd == 'r2':
        command = ['tools.fast_adapter.p29_hammer_candidate_r2', *command]
    elif candidate:
        command = ['tools.fast_adapter.p29_candidate_v04', '--candidate', str(candidate), *command]
    elif rnd == 'r2':
        command = ['tools.fast_adapter.p29_candidate_r2', *command]
    value = {'time': time.time(), 'task': task, 'phase': 'final', 'round': rnd,
             'selection_candidates': candidates, 'selection_rule': 'max OOD correct rate across two fixed seeds; fewer canonical harms; earlier round',
             'selected': chosen, 'base_candidate': str(candidate) if candidate else None,
             'final_outcomes_unopened': True, 'no_future_tuning_on_final': True,
             'conditions': {'canonical': 32, 'train': 64, 'geometry_ood': 64},
             'exact_seeds': strata, 'module_command_without_stratum': command,
             'query_episodes': 1760, 'support_episodes': 320,
             'primary_family': 6, 'statistics': 'unchanged ANALYSIS_FREEZE; family shared across both tasks',
             'bindings': list(bindings.values())}
    save_json(root/'FINAL_FREEZE.json', value)
    print(json.dumps({'task': task, 'round': rnd, 'candidates': candidates, 'command': command}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--rounds', nargs='+', required=True); args = p.parse_args()
    freeze_task(args.task, args.rounds)
