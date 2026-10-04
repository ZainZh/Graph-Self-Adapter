#!/usr/bin/env python3
"""Read-only audit of the frozen 42-cell P54 full-task queue."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_episode(episode, receipt, cell, binding_hash, qualification_hash):
    require(receipt.get('attempt') == 1 and receipt.get('automatic_retry') is False,
            'Receipt does not describe one unretried attempt')
    require(receipt.get('execution_status') == 'finished' and receipt.get('returncode') == 0,
            'Episode process did not exit successfully')
    require(not (episode/'ABORT.json').exists(), 'Episode has ABORT.json')
    artifacts = receipt.get('artifacts', {})
    for name in ('RESULT.json', 'SEAL.json'):
        require(name in artifacts and (episode/name).is_file(), 'Missing '+name)
        require(sha(episode/name) == artifacts[name]['sha256'], 'Receipt hash mismatch: '+name)
    seal = read(episode/'SEAL.json')
    result = read(episode/'RESULT.json')
    require(result.get('cell') == cell and seal.get('cell') == cell,
            'Cell changed in result or seal')
    require(result.get('seal_sha256') == sha(episode/'SEAL.json'), 'Result seal hash mismatch')
    require(seal.get('binding_sha256') == binding_hash == sha(episode/'BINDING.json'),
            'Binding hash mismatch')
    require(seal.get('qualification_sha256') == qualification_hash,
            'Zero-step qualification hash mismatch')
    require(seal.get('original_initial_restores') == 1 and seal.get('later_state_restores') == 0,
            'Restore contract changed')
    require(seal.get('network_before_sha256') == seal.get('network_after_sha256') and
            result.get('network_weights_unchanged') is True,
            'Frozen network hash changed')
    require(result.get('automatic_retry') is False and result.get('independent_statistical_evidence') is False,
            'Scope flags changed')
    require(read(episode/'CELL.json') == cell, 'CELL.json changed')
    for relative, digest in seal.get('files', {}).items():
        path = episode/relative
        require(path.is_file() and path.resolve().is_relative_to(episode.resolve()),
                'Missing or unsafe sealed file: '+relative)
        require(sha(path) == digest, 'Sealed file hash mismatch: '+relative)
    require(seal.get('files'), 'Empty seal')
    prefix, code, micro = (result.get(k) for k in ('prefix_actions', 'code_actions', 'actions'))
    for name, value, limit in (('prefix', prefix, 400), ('code', code, 120), ('micro', micro, 24)):
        require(type(value) is int and 0 <= value <= limit, 'Invalid '+name+' action count')
    require(result.get('total_actions') == prefix+code+micro and result['total_actions'] <= 544,
            'Total action budget mismatch')
    require(type(result.get('updates')) is int and 0 <= result['updates'] <= micro,
            'Invalid z update count')
    require(result.get('near_entry_qualified') is True or micro == 0,
            'Micro actions without Code handoff')
    require(result.get('graph_complete') is True or result.get('failure_stage') in
            {f's{i}' for i in range(1, 10)} | {'prefix_unresolved'},
            'Missing exact failure stage')
    require(not result.get('graph_complete') or
            (result.get('official_success') is True and result.get('near_entry_qualified') is True),
            'Graph completion lacks official goal or s9 entry')
    if cell['arm'] in ('bc', 'fixed'):
        require(result['updates'] == 0, 'BC/fixed z updated')
    actions_path = episode/'public/ACTIONS.jsonl'
    if result.get('near_entry_qualified'):
        require(actions_path.is_file(), 's9 entered without neural action trace')
        rows = [json.loads(line) for line in actions_path.read_text().splitlines()]
        require(len(rows) == micro+1, 'Neural trace/action alignment mismatch')
        require(all(r.get('stage_id') == 's9' and r.get('backend') == 'frozen_stage_policy'
                    for r in rows), 'Neural trace route changed')
        updates = [r.get('updates') for r in rows]
        require(updates == sorted(updates) and updates[-1] == result['updates'],
                'Neural update trace mismatch')
        require([r.get('t') for r in rows] == list(range(micro+1)) and
                all(r.get('executed') is True for r in rows[:-1]) and rows[-1].get('executed') is False,
                'Neural trace does not match executed interactions')
        require(all(type(value) is int and 0 <= value <= i for i, value in enumerate(updates)),
                'z update without a preceding interaction')
        require(all(updates[i]-updates[i-1] in (0, 1) and
                    (updates[i] == updates[i-1] or rows[i-1].get('executed') is True)
                    for i in range(1, len(rows))),
                'z update cannot be linked to an executed prior action')
        judgments = [json.loads(line) for line in (episode/'private/JUDGMENTS.jsonl').read_text().splitlines()]
        require(len(judgments) == len(rows), 'Private judgment/action alignment mismatch')
        if result['graph_complete']:
            final = result.get('final_judgment') or {}
            require(final.get('complete') is True and final.get('goal_hold', 0) >= 3 and
                    final.get('attachment') == 'attached_bilateral',
                    'Graph completion lacks goal/contact hold')
            require(len(judgments) >= 3 and all(j.get('official_success') is True and
                    j.get('finger_contacts') == [True, True] and j.get('wrong_target') is False
                    for j in judgments[-3:]),
                    'Final three goal/contact judgments changed')
    else:
        require(not actions_path.exists() and result['updates'] == 0,
                'z updated outside s9')
    for stage in ('prefix', 'code'):
        trace = episode/stage/'TRACE.jsonl'
        if trace.exists():
            require(all(not ({'z_before', 'z_after', 'updates'} & json.loads(line).keys())
                        for line in trace.read_text().splitlines()),
                    'z trace appeared before s9')
    return dict(index=cell['index'], scale=cell['scale'], condition=cell['condition'],
        arm=cell['arm'], graph_success=bool(result['graph_complete']),
        official_success=bool(result['official_success']), micro_entered=bool(result['near_entry_qualified']),
        failure_stage=result.get('failure_stage'), prefix_actions=prefix,
        code_actions=code, micro_actions=micro, total_actions=result['total_actions'],
        updates=result['updates'], result_sha256=sha(episode/'RESULT.json'),
        seal_sha256=sha(episode/'SEAL.json'))


def span(values):
    return None if not values else [min(values), max(values)]


def summarize(rows, scales=(1.0, 0.9), arms=('bc', 'fixed', 'online')):
    by_key = {(r['scale'], r['condition'], r['arm']): r for r in rows}
    groups = {}
    for scale in scales:
        for arm in arms:
            group = [r for r in rows if r['scale'] == scale and r['arm'] == arm]
            groups[f'{scale:.2f}/{arm}'] = dict(planned=7, completed=len(group),
                graph_success=sum(r['graph_success'] for r in group),
                official_success=sum(r['official_success'] for r in group),
                micro_entered=sum(r['micro_entered'] for r in group),
                failure_stages=dict(sorted(Counter(r['failure_stage'] for r in group if not r['graph_success']).items())),
                action_ranges={k: span([r[k] for r in group]) for k in
                    ('prefix_actions', 'code_actions', 'micro_actions', 'total_actions')},
                update_range=span([r['updates'] for r in group]))
    conditions = sorted({r['condition'] for r in rows})
    fixed_online = {}
    for scale in scales:
        pairs = [(by_key[(scale, c, 'fixed')], by_key[(scale, c, 'online')])
                 for c in conditions if (scale, c, 'fixed') in by_key and (scale, c, 'online') in by_key]
        fixed_online[f'{scale:.2f}'] = dict(paired=len(pairs), planned=7,
            rescued=[a['condition'] for a, b in pairs if not a['graph_success'] and b['graph_success']],
            harmed=[a['condition'] for a, b in pairs if a['graph_success'] and not b['graph_success']],
            both_success=[a['condition'] for a, b in pairs if a['graph_success'] and b['graph_success']],
            both_failure=[a['condition'] for a, b in pairs if not a['graph_success'] and not b['graph_success']])
    size_pairs = {}
    for arm in arms:
        pairs = [(by_key[(1.0, c, arm)], by_key[(0.9, c, arm)])
                 for c in conditions if (1.0, c, arm) in by_key and (0.9, c, arm) in by_key]
        size_pairs[arm] = dict(paired=len(pairs), planned=7,
            smaller_rescued=[a['condition'] for a, b in pairs if not a['graph_success'] and b['graph_success']],
            smaller_harmed=[a['condition'] for a, b in pairs if a['graph_success'] and not b['graph_success']],
            both_success=[a['condition'] for a, b in pairs if a['graph_success'] and b['graph_success']],
            both_failure=[a['condition'] for a, b in pairs if not a['graph_success'] and not b['graph_success']])
    return dict(groups=groups, fixed_online=fixed_online, size_pairs=size_pairs)


def audit(binding_path, qualification_path, queue_dir):
    binding_path, qualification_path, queue_dir = map(Path, (binding_path, qualification_path, queue_dir))
    binding, qualification = read(binding_path), read(qualification_path)
    require(binding.get('schema') == 'prove.p54-book-scale-full-task.v1' and len(binding.get('cells', [])) == 42,
            'Wrong P54 binding scope')
    require(qualification.get('passed') is True and qualification.get('binding_sha256') == sha(binding_path),
            'P54 zero-step qualification invalid')
    require(qualification.get('physics_steps') == 0, 'Qualification ran physics')
    parent = read(binding['parent']['path'])
    require(sha(binding['parent']['path']) == binding['parent']['sha256'], 'P53 parent changed')
    from tools.task_graph.p53_full_task import verify as verify_p53
    verify_p53(parent)
    require(parent['checkpoints']['fixed'] == parent['checkpoints']['online'],
            'Fixed/online checkpoints differ')
    for item in (parent['checkpoints']['bc'], parent['checkpoints']['fixed']):
        require(sha(item['path']) == item['sha256'], 'Frozen P53 checkpoint changed')
    for path, digest in binding['source_files'].items():
        require(sha(path) == digest, 'P54 execution source changed: '+path)
    queue = read(queue_dir/'QUEUE.json')
    queue_result_path = queue_dir/'QUEUE_RESULT.json'
    require(queue_result_path.is_file(), 'Queue has not closed')
    queue_result = read(queue_result_path)
    require(queue_result.get('expected_jobs') == 42 and len(queue_result.get('attempts', [])) == 42 and
            queue_result.get('automatic_retries') == 0,
            'Closed queue does not contain exactly 42 attempts')
    require(queue.get('queue_source_sha256') == sha(Path(__file__).with_name('p53_queue.py')),
            'Queue implementation changed')
    manifest = queue['manifest']
    require(manifest.get('schema') == 'p53.explicit-queue.v1' and manifest.get('attempts_per_job') == 1 and
            manifest.get('workers') == 3 and len(manifest.get('jobs', [])) == 42,
            'P54 queue scope changed')
    require(all(sha(path) == digest for path, digest in manifest['bound_files'].items()) and
            manifest['bound_files'][str(binding_path.resolve())] == sha(binding_path) and
            manifest['bound_files'][str(qualification_path.resolve())] == sha(qualification_path),
            'Queue bound files changed')
    require(len({job['id'] for job in manifest['jobs']}) == 42,
            'Duplicate queue IDs')
    require({x['job']['id'] for x in queue_result['attempts']} == {x['id'] for x in manifest['jobs']},
            'Closed queue attempt IDs differ from manifest')
    completed, engineering_abort, unfinished = [], [], []
    for cell in binding['cells']:
        jobs = [job for job in manifest['jobs'] if '--index' in job['argv'] and
                job['argv'][job['argv'].index('--index')+1] == str(cell['index'])]
        require(len(jobs) == 1, 'Queue index mismatch')
        job = jobs[0]
        for option, expected in (('--binding', binding_path.resolve()),
                                 ('--qualification', qualification_path.resolve())):
            require(option in job['argv'] and Path(job['argv'][job['argv'].index(option)+1]).resolve() == expected,
                    'Queue job reads a different '+option)
        folder = queue_dir/job['id']
        receipt_path = folder/'RECEIPT.json'
        if not receipt_path.is_file():
            unfinished.append(dict(index=cell['index'], id=job['id'], reason='receipt_missing'))
            continue
        try:
            receipt = read(receipt_path)
            require(receipt['job'] == job, 'Receipt job changed')
            require(receipt.get('manifest_sha256') == queue.get('manifest_sha256'),
                    'Receipt belongs to a different queue manifest')
            require(receipt.get('log_sha256') == sha(folder/'process.log'),
                    'Receipt process log hash mismatch')
            completed.append(check_episode(folder/'episode', receipt, cell,
                sha(binding_path), sha(qualification_path)))
        except (ValueError, KeyError, IndexError, OSError, json.JSONDecodeError) as exc:
            engineering_abort.append(dict(index=cell['index'], id=job['id'], reason=str(exc),
                receipt_sha256=sha(receipt_path)))
    stats = summarize(completed)
    status = 'complete' if len(completed) == 42 and not engineering_abort and not unfinished else 'incomplete'
    return dict(schema='prove.p54-full-task-root-audit.v1', status=status,
        binding_sha256=sha(binding_path), qualification_sha256=sha(qualification_path),
        queue_sha256=sha(queue_dir/'QUEUE.json'), planned=42,
        scientifically_completed=len(completed), engineering_abort_count=len(engineering_abort),
        unfinished_count=len(unfinished), engineering_aborts=engineering_abort,
        unfinished=unfinished, statistics=stats, cells=completed,
        rule='Only sealed, validated RESULTs enter success denominators; engineering aborts and missing receipts are separate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.binding, args.qualification, args.queue)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'planned', 'scientifically_completed',
                                           'engineering_abort_count', 'unfinished_count')}, indent=2))
