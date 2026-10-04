#!/usr/bin/env python3
"""Read-only closed-cohort audit for the P55 public s8a graph revision."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

from prove.task_graph.p55_relation_approach import PublicRelationApproach
from tools.task_graph.p54_audit import summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sealed(episode, expected_binding, expected_cell):
    seal = read(episode/'SEAL.json')
    need(seal.get('binding_sha256') == expected_binding == sha(episode/'BINDING.json'),
         'Episode binding hash changed')
    need(seal.get('cell') == expected_cell == read(episode/'CELL.json'), 'Episode cell changed')
    need(seal.get('original_initial_restores') == 1 and seal.get('later_state_restores') == 0,
         'Original-state restore contract changed')
    for name, digest in seal.get('files', {}).items():
        path = episode/name
        need(path.resolve().is_relative_to(episode.resolve()) and path.is_file() and sha(path) == digest,
             'Missing/changed sealed file: '+name)
    need(bool(seal.get('files')), 'Empty episode seal')
    return seal


def qualification(binding_path, qualification_path):
    q = read(qualification_path)
    need(q.get('schema') == 'prove.p55-two-nominal-real-prefix-code-qualification.v1' and q.get('passed') is True,
         'P55 real Code qualification not passed')
    need(q.get('binding_sha256') == sha(binding_path) and q.get('scale_count') == 2 and
         q.get('automatic_retries') == 0 and q.get('teacher_executed') is False and
         q.get('neural_executed') is False, 'P55 qualification contract changed')
    for scale in (1.0, .9):
        folder = qualification_path.parent/f'scale_{scale:.2f}'
        result = read(folder/'RESULT.json')
        need(result in q['outcomes'] and result['scale'] == scale and result['condition'] == 'nominal' and
             result['prefix_qualified'] and result['near_entry_qualified'] and
             not result['official_success_at_stop'] and result['micro_actions'] == 0 and
             result['teacher_actions'] == 0 and result['automatic_retry'] is False,
             'Nominal real Code qualification changed')
        need(result['arm'] == 'bc', 'Nominal qualification arm changed')
        sealed(folder, sha(binding_path), read(folder/'CELL.json'))
        need(result['seal_sha256'] == sha(folder/'SEAL.json'), 'Qualification seal mismatch')
    return q


def closure(queue, binding_path, qualification_path):
    result_path = queue/'QUEUE_RESULT.json'
    need(result_path.is_file(), 'Queue is not closed')
    result = read(result_path)
    need(result.get('expected_jobs') == 42 and len(result.get('attempts', [])) == 42 and
         result.get('automatic_retries') == 0, 'Closed queue lacks 42 attempts')
    frozen = read(queue/'QUEUE.json')
    plan = frozen['manifest']
    need(plan.get('schema') == 'p53.explicit-queue.v1' and plan.get('attempts_per_job') == 1 and
         plan.get('workers') == 3 and len(plan.get('jobs', [])) == 42,
         'Queue plan changed')
    need(frozen.get('queue_source_sha256') == sha(Path(__file__).with_name('p53_queue.py')),
         'Queue source changed')
    need(all(sha(path) == digest for path, digest in plan['bound_files'].items()),
         'Queue bound source or input changed')
    need(plan['bound_files'][str(binding_path.resolve())] == sha(binding_path) and
         plan['bound_files'][str(qualification_path.resolve())] == sha(qualification_path),
         'Queue points at another freeze or qualification')
    need({x['job']['id'] for x in result['attempts']} == {x['id'] for x in plan['jobs']},
         'Closed queue job IDs differ')
    return frozen, plan


def code_route(episode, result, reference41, reference57):
    path = episode/'code/TRACE.jsonl'
    if not path.exists():
        need(result['code_actions'] == 0 and not result['near_entry_qualified'],
             'Code actions lack trace')
        return dict(s8a_actions=0, reached_s8a=False, transitions=[],
                    public_replay_max_action_error=0.0)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    need(len(rows) == result['code_actions']+1 and [x['t'] for x in rows] == list(range(len(rows))),
         'Code trace/action alignment changed')
    need(sum(x['action'] is not None for x in rows) == result['code_actions'] and
         rows[-1]['action'] is None and result['code_actions'] <= 120,
         'Code action budget changed')
    allowed = ('s7','s8','s8a')
    seen = [x['stage_id'] for x in rows]
    need(all(s in allowed for s in seen) and all(allowed.index(b) >= allowed.index(a)
         for a,b in zip(seen, seen[1:])), 'Code stages reversed or unknown')
    transitions = [dict(**t, after_actions=x['t']) for x in rows for t in x['transitions']]
    targets = read(episode/'code/PUBLIC_TARGETS.json')['targets']
    need(targets[1] == reference41 and targets[2] == reference57,
         'Public Code references 41/57 changed')
    controller = PublicRelationApproach(targets[1], targets[2])
    replay_errors = []
    if 's8a' in seen:
        need(any(t['from_stage'] == 's8' and t['to_stage'] == 's8a' for t in transitions),
             'No real s8 -> s8a transition')
    if result['near_entry_qualified']:
        need(seen[-1] == 's8a' and rows[-1]['stop_reason'] == 'near_ready' and
             rows[-1]['supervisor_gate']['near_ready'] is True,
             's9 entry lacks original near-ready gate')
    for row in rows:
        need(not ({'z_before','z_after','updates'} & row.keys()),
             'z update appeared in Code')
        if row['stage_id'] == 's8a':
            diagnostic = row['public_controller']
            need(all(key in diagnostic for key in ('direction','public_eef_position',
                'public_eef_quaternion_xyzw','lateral_error','rotation_error')),
                's8a lacks public EEF diagnostic')
            if row['action'] is not None:
                replay, _ = controller.command(diagnostic['public_eef_position'],
                                               diagnostic['public_eef_quaternion_xyzw'])
                error = float(np.max(np.abs(replay-np.asarray(row['action'], dtype=float))))
                need(np.isfinite(error) and error <= 1e-12,
                     's8a action cannot be recomputed from public EEF and references')
                replay_errors.append(error)
    return dict(s8a_actions=sum(x['stage_id'] == 's8a' and x['action'] is not None for x in rows),
        reached_s8a='s8a' in seen, transitions=transitions,
        public_replay_max_action_error=max(replay_errors, default=0.0))


def episode_result(episode, receipt, cell, binding_hash, qualification_hash, reference41, reference57):
    need(receipt.get('attempt') == 1 and receipt.get('automatic_retry') is False and
         receipt.get('execution_status') == 'finished' and receipt.get('returncode') == 0,
         'Episode launch aborted')
    need(not (episode/'ABORT.json').exists(), 'Episode ABORT.json exists')
    for name in ('RESULT.json','SEAL.json'):
        need(name in receipt.get('artifacts', {}) and
             sha(episode/name) == receipt['artifacts'][name]['sha256'],
             'Receipt artifact changed: '+name)
    seal = sealed(episode, binding_hash, cell)
    result = read(episode/'RESULT.json')
    need(result.get('cell') == cell and result.get('seal_sha256') == sha(episode/'SEAL.json') and
         result.get('qualification_sha256') == qualification_hash,
         'Result freeze or qualification changed')
    need(seal.get('network_before_sha256') == seal.get('network_after_sha256') and
         result.get('network_weights_unchanged') is True and result.get('automatic_retry') is False,
         'Frozen model or retry contract changed')
    for key, limit in (('prefix_actions',400),('code_actions',120),('actions',24)):
        need(type(result.get(key)) is int and 0 <= result[key] <= limit,
             'Action budget changed: '+key)
    need(result.get('total_actions') == result['prefix_actions']+result['code_actions']+result['actions']
         and result['total_actions'] <= 544, 'Total action count changed')
    need(type(result.get('updates')) is int and 0 <= result['updates'] <= result['actions'],
         'Invalid z update count')
    if cell['arm'] != 'online' or not result['near_entry_qualified']:
        need(result['updates'] == 0, 'z update outside online s9')
    route = code_route(episode, result, reference41, reference57)
    if result['near_entry_qualified']:
        need(route['reached_s8a'] and (episode/'public/ACTIONS.jsonl').exists(),
             's9 entered without s8a or neural trace')
        transitions = read(episode/'STAGE_TRANSITIONS.json')
        need(any(t['from_stage'] == 's8a' and t['to_stage'] == 's9' for t in transitions),
             'No s8a -> s9 graph transition')
        actions = [json.loads(line) for line in (episode/'public/ACTIONS.jsonl').read_text().splitlines()]
        need(len(actions) == result['actions']+1 and all(x['stage_id'] == 's9' for x in actions) and
             actions[-1]['updates'] == result['updates'] and
             [x['t'] for x in actions] == list(range(len(actions))) and
             all(x.get('executed') is True for x in actions[:-1]) and
             actions[-1].get('executed') is False and
             all(type(x['updates']) is int and 0 <= x['updates'] <= x['t'] for x in actions) and
             all(actions[i]['updates']-actions[i-1]['updates'] in (0,1)
                 for i in range(1,len(actions))) and
             (cell['arm'] == 'online' or all(x['updates'] == 0 for x in actions)),
             'Neural action/update trace changed')
    else:
        need(result['actions'] == 0 and result['updates'] == 0 and
             not (episode/'public/ACTIONS.jsonl').exists(),
             'Neural action/update without s9 entry')
    if result['graph_complete']:
        final = result.get('final_judgment') or {}
        need(result['official_success'] is True and result['near_entry_qualified'] and
             final.get('complete') is True and final.get('goal_hold',0) >= 3 and
             final.get('attachment') == 'attached_bilateral',
             'Graph completion lacks native goal/contact3')
        judgments = [json.loads(x) for x in (episode/'private/JUDGMENTS.jsonl').read_text().splitlines()]
        need(len(judgments) >= 3 and all(j['official_success'] and
             j['finger_contacts'] == [True,True] and not j['wrong_target'] for j in judgments[-3:]),
             'Final three judgments fail goal/contact')
    if read(episode/'CODE_RESULT.json').get('completed_goal_before_s9'):
        need(not result['graph_complete'] and not result['near_entry_qualified'],
             'Code-stage native goal incorrectly counted as graph success')
    return dict(index=cell['index'], scale=cell['scale'], condition=cell['condition'], arm=cell['arm'],
        graph_success=bool(result['graph_complete']), official_success=bool(result['official_success']),
        micro_entered=bool(result['near_entry_qualified']), failure_stage=result.get('failure_stage'),
        prefix_actions=result['prefix_actions'], code_actions=result['code_actions'],
        s8a_actions=route['s8a_actions'], micro_actions=result['actions'],
        public_replay_max_action_error=route['public_replay_max_action_error'],
        total_actions=result['total_actions'], updates=result['updates'],
        code_stage_goal_before_s9=bool(read(episode/'CODE_RESULT.json').get('completed_goal_before_s9')),
        prefix_hashes={k:v for k,v in seal['files'].items() if k.startswith('prefix/')},
        result_sha256=sha(episode/'RESULT.json'), seal_sha256=sha(episode/'SEAL.json'))


def audit(binding_path, qualification_path, queue, p54_queue):
    binding_path, qualification_path, queue, p54_queue = map(Path,
        (binding_path, qualification_path, queue, p54_queue))
    _, plan = closure(queue, binding_path, qualification_path)
    binding = read(binding_path)
    from tools.task_graph.p55_relation_approach import verify as verify_p55
    verify_p55(binding)
    need(binding.get('schema') == 'prove.p55-public-relation-approach.v1' and
         len(binding.get('cells', [])) == 42 and binding.get('automatic_retries') == 0,
         'P55 binding scope changed')
    parent = read(binding['parent']['path'])
    need(sha(binding['parent']['path']) == binding['parent']['sha256'], 'P54 parent changed')
    need(parent['cells'] == binding['cells'] and parent['schema'] == 'prove.p54-book-scale-full-task.v1',
         'P54 cell pairing changed')
    p53 = read(parent['parent']['path'])
    qualification(binding_path, qualification_path)
    old_queue = read(p54_queue/'QUEUE.json')
    need((p54_queue/'QUEUE_RESULT.json').is_file() and
         read(p54_queue/'QUEUE_RESULT.json').get('expected_jobs') == 42,
         'P54 comparison queue not closed')
    old_jobs = {int(j['argv'][j['argv'].index('--index')+1]): j for j in old_queue['manifest']['jobs']}
    jobs = {int(j['argv'][j['argv'].index('--index')+1]): j for j in plan['jobs']}
    need(set(jobs) == set(old_jobs) == set(range(42)), '42 paired cell IDs differ')
    rows, old_rows, aborts = [], [], []
    for cell in binding['cells']:
        i = cell['index']
        folder = queue/jobs[i]['id']
        receipt_path = folder/'RECEIPT.json'
        if not receipt_path.exists():
            aborts.append(dict(index=i, reason='receipt_missing'))
            continue
        try:
            receipt = read(receipt_path)
            need(receipt['job'] == jobs[i] and receipt['manifest_sha256'] == read(queue/'QUEUE.json')['manifest_sha256'] and
                 receipt['log_sha256'] == sha(folder/'process.log'),
                 'Queue receipt or log changed')
            row = episode_result(folder/'episode', receipt, cell, sha(binding_path),
                                 sha(qualification_path), p53['reference_poses'][41],
                                 p53['reference_poses'][57])
            old_folder = p54_queue/old_jobs[i]['id']
            old_episode = old_folder/'episode'
            old_receipt = read(old_folder/'RECEIPT.json')
            need(old_receipt.get('job') == old_jobs[i] and all(
                name in old_receipt.get('artifacts', {}) and
                sha(old_episode/name) == old_receipt['artifacts'][name]['sha256']
                for name in ('RESULT.json','SEAL.json')),
                'P54 paired RESULT/SEAL receipt hash changed')
            old_result, old_seal = read(old_episode/'RESULT.json'), read(old_episode/'SEAL.json')
            need(old_result['cell'] == cell and old_result['seal_sha256'] == sha(old_episode/'SEAL.json'),
                 'P54 paired baseline changed')
            old_prefix = {k:v for k,v in old_seal['files'].items() if k.startswith('prefix/')}
            need(row['prefix_hashes'] == old_prefix, 'P55 prefix differs from P54 in paired cell')
            old_rows.append(dict(index=i, scale=cell['scale'], condition=cell['condition'], arm=cell['arm'],
                graph_success=bool(old_result['graph_complete']), official_success=bool(old_result['official_success'])))
            del row['prefix_hashes']
            rows.append(row)
        except (ValueError, KeyError, IndexError, OSError, json.JSONDecodeError) as exc:
            aborts.append(dict(index=i, reason=str(exc), receipt_sha256=sha(receipt_path)))
    old_by_index = {x['index']:x for x in old_rows}
    gains = {}
    for scale in (1.0,.9):
        for arm in ('bc','fixed','online'):
            group = [r for r in rows if r['scale'] == scale and r['arm'] == arm]
            pairs = [(old_by_index[r['index']],r) for r in group]
            gains[f'{scale:.2f}/{arm}'] = dict(planned=7,paired=len(pairs),
                rescued=[new['condition'] for old,new in pairs if not old['graph_success'] and new['graph_success']],
                harmed=[new['condition'] for old,new in pairs if old['graph_success'] and not new['graph_success']],
                both_success=[new['condition'] for old,new in pairs if old['graph_success'] and new['graph_success']],
                both_failure=[new['condition'] for old,new in pairs if not old['graph_success'] and not new['graph_success']])
    status = 'complete' if len(rows) == 42 and not aborts else 'incomplete'
    stats = summarize(rows)
    for scale in (1.0,.9):
        for arm in ('bc','fixed','online'):
            group = [r for r in rows if r['scale'] == scale and r['arm'] == arm]
            item = stats['groups'][f'{scale:.2f}/{arm}']
            item['s8a_action_range'] = None if not group else [min(r['s8a_actions'] for r in group),
                                                               max(r['s8a_actions'] for r in group)]
            item['code_stage_official_secondary'] = sum(r['code_stage_goal_before_s9'] for r in group)
    return dict(schema='prove.p55-full-task-root-audit.v1', status=status, planned=42,
        scientifically_completed=len(rows), engineering_abort_count=len(aborts), engineering_aborts=aborts,
        binding_sha256=sha(binding_path), qualification_sha256=sha(qualification_path),
        queue_sha256=sha(queue/'QUEUE.json'), p54_queue_sha256=sha(p54_queue/'QUEUE.json'),
        statistics=stats, p54_to_p55_graph_pairs=gains,
        code_stage_official_secondary=sum(x['code_stage_goal_before_s9'] for x in rows),
        s8a_action_range=None if not rows else [min(x['s8a_actions'] for x in rows),
                                                   max(x['s8a_actions'] for x in rows)],
        public_s8a_replay_max_action_error=max((x['public_replay_max_action_error'] for x in rows), default=0.0),
        cells=rows, rule='Only validated sealed episodes enter denominators; Code-stage official success is secondary, never graph completion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binding','qualification','queue','p54-queue','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.binding,args.qualification,args.queue,args.p54_queue)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','planned','scientifically_completed',
                                                 'engineering_abort_count')},indent=2))
