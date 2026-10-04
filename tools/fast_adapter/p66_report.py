#!/usr/bin/env python3
"""P66 postseal report: four arms, exact opened-development denominator.

This module never runs a simulator. No outcome may be opened until all 48
predeclared cells have both result and seal files and no program aborts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np

from tools.fast_adapter.p65_report import PHASES, need, read, ref, replay_neural_commands

ARMS = ('frame_fixed', 'reference_fixed', 'entry_once', 'continuous')
SPLITS = ('id', 'size', 'position')
SEEDS = tuple(range(640100, 640104))
COMPARISONS = {
    'B-A': ('frame_fixed', 'reference_fixed', 'reference holding; no gradient benefit claim'),
    'D-B': ('reference_fixed', 'continuous', 'entry initialization plus continuing feedback'),
    'C-B': ('reference_fixed', 'entry_once', 'one eligible entry initialization'),
    'D-C': ('entry_once', 'continuous', 'incremental continuing feedback'),
    'C-A': ('frame_fixed', 'entry_once', 'entry initialization versus original frame baseline'),
    'D-A': ('frame_fixed', 'continuous', 'continuing feedback versus original frame baseline'),
}


def status(directory):
    """Inspect filenames only. Never open a partial result or seal."""
    root = Path(directory)
    results = {p.parent for p in root.glob('*/RESULT.json')}
    seals = {p.parent for p in root.glob('*/SEAL.json')}
    aborts = list(root.glob('*/ABORT.json')) + list(root.glob('*/PROGRAM_ERROR.json'))
    return dict(expected=48, result_files=len(results), seal_files=len(seals), aborts=len(aborts),
                complete=len(results) == 48 and results == seals and not aborts)


def require_complete(directory):
    receipt = status(directory)
    need(receipt['complete'], 'Require all 48 paired sealed cells and zero aborts before opening outcomes')
    return receipt


def key(result):
    cell = result['cell']
    return cell['split'], cell['seed'], cell['arm']


def aggregate(runs):
    lookup = {key(r['result']): r for r in runs}
    expected = {(s, seed, arm) for s in SPLITS for seed in SEEDS for arm in ARMS}
    need(len(runs) == 48 and set(lookup) == expected, 'Exact 48-cell condition/arm denominator required')
    groups, pairs, comparisons = {}, [], {}
    for split in SPLITS:
        for arm in ARMS:
            selected = [lookup[split, seed, arm] for seed in SEEDS]
            groups[split+'/'+arm] = dict(
                n=4, success=sum(bool(r['result']['validated_success']) for r in selected),
                sustained_native_success=sum(bool(r['result']['sustained_native_success']) for r in selected),
                mechanical_anomalies=sum(not r['result']['mechanical_valid'] for r in selected),
                phase_entered={p: sum(p in r['phase_first_t'] for r in selected) for p in PHASES},
                phase_steps={p: sum(r['phase_steps'].get(p, 0) for r in selected) for p in PHASES},
                terminal_phases=dict(Counter(r['terminal_phase'] for r in selected)),
                did_not_reach_close=sum('close' not in r['phase_first_t'] for r in selected),
                nn_command_steps=sum(r['nn_command_steps'] for r in selected),
                code_command_steps=sum(r['code_command_steps'] for r in selected),
                update_events=sum(r['update_events'] for r in selected))
        for label, (base, candidate, interpretation) in COMPARISONS.items():
            selected = []
            for seed in SEEDS:
                a = bool(lookup[split, seed, base]['result']['validated_success'])
                b = bool(lookup[split, seed, candidate]['result']['validated_success'])
                row = dict(split=split, seed=seed, comparison=label, baseline=base, candidate=candidate,
                           baseline_success=a, candidate_success=b, rescue=b and not a, harm=a and not b)
                selected.append(row)
                pairs.append(row)
            comparisons[split+'/'+label] = dict(n=4, rescues=sum(p['rescue'] for p in selected),
                harms=sum(p['harm'] for p in selected), interpretation=interpretation)
    return groups, pairs, comparisons


def same(a, b, message):
    need(np.shape(a) == np.shape(b) and np.allclose(a, b, atol=1e-9, rtol=0, equal_nan=False), message)


def audit_prefix(first, other, stop):
    """Compare actual saved commands, simulator state and public inputs before stop."""
    need(isinstance(stop, int) and 0 <= stop <= 240, 'Invalid prefix boundary')
    need(len(first['trace']) >= stop and len(other['trace']) >= stop, 'Incomplete prefix trace')
    for i in range(stop):
        a, b = first['trace'][i], other['trace'][i]
        need(all(a[k] == b[k] for k in ('t', 'phase_before', 'phase_after', 'closed', 'command_mode')),
             'Common-prefix routing differs')
        for name in ('command_p', 'command_R', 'controls_before', 'controls_after'):
            same(a[name], b[name], 'Common-prefix command differs: '+name)
        for name in ('integration', 'qpos'):
            same(a['state'][name], b['state'][name], 'Common-prefix simulator state differs')
        oa, ob = first['observations'][i], other['observations'][i]
        need(set(oa) == set(ob), 'Common-prefix public fields differ')
        for name in oa:
            same(oa[name], ob[name], 'Common-prefix public observation differs')
    samples = None
    if 'path' in first and 'path' in other:
        with np.load(Path(first['path'])/'PUBLIC_IMAGES.npz') as a, np.load(Path(other['path'])/'PUBLIC_IMAGES.npz') as b:
            need(np.array_equal(a['t'], b['t']), 'Public image schedule differs')
            take = a['t'] < stop
            for name in ('rgb', 'masks_packed'):
                need(np.array_equal(a[name][take], b[name][take]), 'Common-prefix public image/mask differs')
            samples = int(take.sum())
    return dict(verified=True, steps=stop, public_image_samples_checked=samples)


def audit_pairing(runs):
    lookup = {key(r['result']): r for r in runs}
    receipts = []
    for split in SPLITS:
        for seed in SEEDS:
            selected = [lookup[split, seed, arm] for arm in ARMS]
            entry = [next((i for i, r in enumerate(run['trace']) if r['phase_before']=='approach'), 240)
                     for run in selected]
            need(len(set(entry)) == 1, 'Four-arm approach entrance differs')
            common = [audit_prefix(selected[0], run, entry[0]) for run in selected[1:]]
            # A step zero comparison is still required if entry is immediate.
            for run in selected[1:]:
                for name in ('integration', 'qpos'):
                    same(selected[0]['trace'][0]['state'][name], run['trace'][0]['state'][name],
                         'Four-arm initial simulator state differs')
                a, b = selected[0]['observations'][0], run['observations'][0]
                need(set(a)==set(b), 'Four-arm initial public fields differ')
                for name in a: same(a[name], b[name], 'Four-arm initial public state differs')
            once, continuous = selected[2:]
            stop = continuous['first_continuation_t']
            stop = 240 if stop is None else stop
            continued = audit_prefix(once, continuous, stop)
            need(once['entry_initialization'] == continuous['entry_initialization'],
                 'Entry-once/continuous initialization inputs or result differ')
            receipts.append(dict(split=split, seed=seed, four_arm_prefix=common,
                                 first_approach_t=entry[0], entry_once_continuous_prefix=continued,
                                 identical_entry_initialization=True))
    return receipts


def audit_reference_records(records, arm):
    """Independently reconstruct reference selection and event eligibility."""
    need(arm in ARMS and records, 'Missing P66 reference records')
    latest = reference = anchor = previous = None
    ref_time = entry_t = continuation_t = None
    entered = initialized = False
    entry_receipt = None
    events = 0
    admitted_times = set()
    for r in records:
        t, phase = r['t'], r['phase_before']
        need(previous is None or t > previous['t'], 'Noncausal record order')
        need(phase not in ('close', 'transport', 'hold'), 'Encoder after critical exit')
        need(r['arm'] == arm, 'Reference arm differs')
        raw = np.asarray(r['raw_geometry8'])
        z0, z1 = np.asarray(r['z_before']), np.asarray(r['z_after'])
        need(raw.shape == z0.shape == z1.shape == (8,) and
             all(np.isfinite(x).all() for x in (raw, z0, z1)), 'Invalid finite geometry/latent8')
        need(np.array_equal(z0[3:6], np.zeros(3)) and np.array_equal(z1[3:6], np.zeros(3)),
             'Orientation latent changed')
        if anchor is None: anchor = np.asarray(r['T_world_anchor'])
        need(np.array_equal(anchor, r['T_world_anchor']), 'Episode anchor rebased')
        support = r['support_times']
        need(len(support) == r['S'] and len(set(support)) == len(support)
             and all(0 <= s <= t for s in support), 'Future or duplicated public support')
        if r['admitted']:
            latest = (t, raw.copy()); admitted_times.add(t)
        need(set(support) <= admitted_times, 'Support was not previously admitted')
        critical = phase == 'approach'
        first = critical and not entered
        if first: entered = True; entry_t = t
        need(bool(r['entry_at_this_frame']) == first and bool(r['entry_seen']) == entered,
             'Invalid approach-entry receipt')
        need(np.array_equal(z0, np.zeros(8) if first or previous is None else previous['z_after']),
             'Latent continuity or entry reset differs')
        if critical and arm != 'frame_fixed' and reference is None and latest is not None:
            ref_time, reference = latest[0], latest[1].copy()
        need(r['ref_time'] == ref_time and bool(r['reference_available']) == (reference is not None),
             'Reference time/availability differs from admitted history')
        if reference is None: need(r['g_ref'] is None, 'Reference fabricated without admitted history')
        else: need(np.array_equal(r['g_ref'], reference), 'Fixed reference changed or differs from admitted readout')
        base = reference if critical and arm != 'frame_fixed' else raw
        need((base is None and r['decoder_base'] is None) or
             (base is not None and np.array_equal(base, r['decoder_base'])), 'Decoder does not use declared base')
        expected = None
        if critical and reference is not None and arm in ('entry_once', 'continuous') and r['S'] >= 3:
            if not initialized: expected = 'entry'; initialized = True
            elif arm == 'continuous' and r['admitted']: expected = 'continuous'
        need(r['event'] == expected and bool(r['entry_initialized']) == initialized,
             'Entry/continuing update eligibility differs')
        need(r['U'] == (3 if expected else 0) and len(r['inner']) == r['U'], 'Wrong inner update budget')
        if expected:
            events += 1
            for inner in r['inner']:
                need(inner['active_dim'] == 5 and inner['active_indices'] == [0,1,2,6,7],
                     'Update includes forbidden orientation components')
                need(all(np.isfinite(inner[k]) and inner[k] >= 0 for k in
                         ('loss', 'gradient_norm', 'curvature_scale', 'delta_norm')), 'Nonfinite gradient receipt')
            if expected == 'entry':
                entry_receipt = dict(t=t, ref_time=ref_time, reference=reference.tolist(),
                    anchor=anchor.tolist(), support_times=list(support), z_before=z0.tolist(),
                    z_after=z1.tolist(), inner=r['inner'])
            elif continuation_t is None: continuation_t = t
        else: need(np.array_equal(z0, z1), 'Latent changed without eligible feedback')
        if not critical or arm in ('frame_fixed', 'reference_fixed'):
            need(np.array_equal(z1, np.zeros(8)), 'Nonzero latent outside update arm/stage')
        qualified = base is not None and (reference is not None if critical and arm != 'frame_fixed' else r['goal_valid'])
        need((r['decoded'] is not None) == bool(qualified), 'Missing-feedback estimate qualification differs')
        if qualified:
            import torch
            from prove.fast_adapter.p61_adaptation import corrected_components
            with torch.no_grad():
                p, rotation, size = corrected_components(torch.as_tensor(base), torch.as_tensor(z1))
                transform = torch.as_tensor(anchor)
                expected_decoded = (transform[:3,:3]@p+transform[:3,3], transform[:3,:3]@rotation, size)
            need(len(r['decoded']) == 3 and all(np.shape(a)==tuple(b.shape) and
                 np.allclose(a,b.numpy(),atol=2e-6,rtol=0) for a,b in zip(r['decoded'],expected_decoded)),
                 'Decoded estimate differs from fixed reference plus latent')
        if previous is not None and critical and arm != 'frame_fixed' and not expected and reference is not None \
                and previous['g_ref'] is not None:
            need(all(np.array_equal(a,b) for a,b in zip(r['decoded'], previous['decoded'])),
                 'Held reference estimate drifted without update')
        previous = r
    return dict(update_events=events, first_approach_t=entry_t, first_continuation_t=continuation_t,
                entry_initialization=entry_receipt, reference_time=ref_time,
                reference_invariants_verified=True)


def audit_trace(trace, observations, arm):
    need(len(trace) == len(observations) == 240, 'P66 requires H240 complete traces')
    phases, occupancy, records = {}, Counter(), []
    nn = 0
    previous = None
    for t, (row, public) in enumerate(zip(trace, observations)):
        need(row['t'] == public['t'] == t, 'Noncontiguous public/control trace')
        need(set(public) == {'t','K','camera_position','camera_rotation','position','rotation'},
             'Nonpublic or missing runtime input')
        before, after = row['phase_before'], row['phase_after']
        need(before in PHASES and after in PHASES, 'Unknown phase')
        if t: need(before == trace[t-1]['phase_after'], 'Phase discontinuity')
        phases.setdefault(before,t); phases.setdefault(after,t); occupancy[before] += 1
        estimate, route = row['estimate'], row['neural_action']
        qualified = estimate is not None and all(np.isfinite(x).all() for x in estimate)
        neural = bool(t >= 12 and before == 'approach' and qualified)
        need(route['used'] == neural and row['command_mode'] == route['route'] == ('neural' if neural else 'code'),
             'Actual neural command escaped approach or differs from route')
        need(route['phase_before'] == before and route['phase_after'] == after
             and bool(route['gripper_closed']) == bool(row['closed']), 'Routing metadata differs')
        same(row['command_p'], route['target'], 'Actual target differs from routing record')
        if neural:
            nn += 1
            need(route['neural_phase'] == 'approach' and all(k in route for k in ('center','size','start','anchor')),
                 'Missing actual neural inputs')
        else:
            same(row['command_p'], route['code_target'], 'Code target overridden')
            same(row['command_R'], route['code_rotation'], 'Code rotation overridden')
        record = row['adaptation']
        need(record is not None, 'Missing adaptation receipt')
        if before in ('close','transport','hold'):
            need(previous is not None and record['t'] == previous['t'] and record['t'] < t,
                 'Encoder updated after close')
            need(np.array_equal(record['z_after'], previous['z_after']), 'Archived postclose latent changed')
        else:
            need(record['t'] == t and record['phase_before'] == before, 'Missing current preclose readout')
            records.append(record); previous = record
        need((record['decoded'] is None) == (estimate is None), 'Saved estimate differs from decoder')
        if estimate is not None:
            for a,b in zip(estimate, record['decoded']):same(a,b,'Actual estimate differs from decoder')
    stats = audit_reference_records(records, arm)
    return dict(phase_first_t=phases,phase_steps=dict(occupancy),terminal_phase=trace[-1]['phase_after'],
                nn_command_steps=nn,code_command_steps=240-nn,**stats)


def audit_lifecycle(trace, lifecycle):
    need(len(lifecycle) == 1, 'Exactly one latent exit required')
    event = lifecycle[0]
    need(event['cached_estimate_retained'] and event['archive_retained']
         and np.array_equal(event['z_after'],np.zeros(8)), 'Exit failed to clear live z and retain cache')
    boundaries = [r for r in trace if r['phase_before']=='approach' and r['phase_after']=='close']
    if boundaries:
        need(len(boundaries)==1 and event['reason']=='approach_to_close' and event['t']==boundaries[0]['t'],
             'Wrong critical exit boundary')
        row = boundaries[0]
        need(row['latent_exit'] is not None and row['latent_exit']['t']==event['t'], 'Missing trace exit receipt')
    else:
        need(event['reason']=='episode_budget_end' and event['t']==240, 'Missing budget-end reset')
        row = trace[-1]
    need(np.array_equal(event['z_before'],row['adaptation']['z_after']), 'Exit latent differs from archived prediction')


def load_cell(path, freeze, freeze_path):
    from prove.fast_adapter.p61_data import sha256
    path = Path(path); result = read(path/'RESULT.json'); seal = read(path/'SEAL.json')
    required = {'TRACE.npy','OBS.npy','TERMINAL.npy','PRE_RUN.json','PUBLIC_IMAGES.npz','LATENT_LIFECYCLE.npy'}
    need(required <= set(seal['files']), 'Incomplete sealed artifacts')
    for name, digest in seal['files'].items():
        target = (path/name).resolve()
        need(target.is_relative_to(path.resolve()) and sha256(target)==digest, 'Changed sealed artifact: '+name)
    pre = read(path/'PRE_RUN.json')
    need(pre['freeze_sha256']==sha256(freeze_path) and pre['runtime_truth'] is False and pre['steps']==240,
         'Pre-run frozen scope differs')
    need(0 <= pre['index'] < 48 and pre['cell']==result['cell']==freeze['cells'][pre['index']], 'Cell identity differs')
    need(seal['weights_before']==seal['weights_after'] and result['weights_unchanged'], 'Frozen weights changed')
    trace = np.load(path/'TRACE.npy',allow_pickle=True).tolist()
    observations = np.load(path/'OBS.npy',allow_pickle=True).tolist()
    stats = audit_trace(trace, observations, result['cell']['arm'])
    audit_lifecycle(trace,np.load(path/'LATENT_LIFECYCLE.npy',allow_pickle=True).tolist())
    for name, value in [('neural_action_steps',stats['nn_command_steps']),('code_action_steps',stats['code_command_steps']),
                        ('update_events',stats['update_events']),('reference_time',stats['reference_time']),
                        ('final_phase',stats['terminal_phase'])]:
        need(result[name]==value,'Result differs from trace: '+name)
    need(result['latent_end_reset'] and result['orientation_ood'] is False, 'Reset/scope receipt differs')
    return dict(path=str(path.resolve()),result=result,trace=trace,observations=observations,
                result_binding=ref(path/'RESULT.json'),seal_binding=ref(path/'SEAL.json'),weights_unchanged=True,**stats)


def report(freeze_path, directory, output, p65_report):
    from prove.fast_adapter.p61_data import sha256
    require_complete(directory)
    freeze = read(freeze_path)
    expected = [dict(split=s,seed=seed,arm=a) for s in SPLITS for seed in SEEDS for a in ARMS]
    need(freeze.get('schema')=='prove.p66-cup-reference-freeze.v1' and freeze.get('status')=='frozen'
         and freeze.get('steps')==240 and freeze.get('cells')==expected, 'Wrong P66 frozen scope')
    for path,digest in freeze['files'].items(): need(sha256(path)==digest,'Frozen file changed: '+path)
    runs = [load_cell(p.parent,freeze,freeze_path) for p in sorted(Path(directory).glob('*/RESULT.json'))]
    groups,pairs,comparisons = aggregate(runs)
    pairing = audit_pairing(runs)
    replay = replay_neural_commands(runs,freeze['action_policy'])
    history = historical_reference(p65_report)
    value = dict(schema='prove.p66-reference-state-cup-complete-audit.v1',complete=True,
        freeze=ref(freeze_path),report_source=ref(__file__),groups=groups,pairs=pairs,comparisons=comparisons,
        all_runtime_invariants_passed=True,pairing=pairing,independent_neural_command_replay=replay,
        historical_p65_reference=history,physics_steps=0,
        definitions=dict(scope='12 opened-development conditions x 4 arms; not independent held-out; no orientation OOD',
            success='unchanged P47 validated_success, sustained_native_success and mechanical_valid',
            attribution='B-A is reference holding, not gradient adaptation; C-A/D-A retain original frame baseline',
            stalls='terminal phases and failure to enter close by H240; descriptive only',
            denominator='all four predefined conditions per split/arm including mechanical failures'),
        cells=[{k:v for k,v in r.items() if k not in ('trace','observations')} for r in runs])
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    (output/'RESULTS.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


def historical_reference(path):
    source = read(path)
    need(source.get('schema') == 'prove.p65-stage-local-cup-complete-audit.v1'
         and source.get('complete') and len(source['cells']) == 36
         and source.get('all_runtime_invariants_passed'), 'Historical reference must be completed P65 report')
    return dict(source=ref(path), groups={s+'/frame_fixed': source['groups'][s+'/fixed'] for s in SPLITS},
                interpretation='Separate historical P65 fixed arm; not pooled, no new physics or randomized arm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--p65-report', type=Path)
    args = parser.parse_args()
    if args.status_only:
        print(json.dumps(status(args.cells)))
        return
    if any(x is None for x in (args.freeze,args.output,args.p65_report)):
        parser.error('--freeze, --output and --p65-report required')
    value=report(args.freeze,args.cells,args.output,args.p65_report)
    print(json.dumps(value['comparisons'],indent=2))


if __name__ == '__main__':
    main()
