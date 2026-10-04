#!/usr/bin/env python3
"""P67 full-cohort, condition-streamed audit and predeclared paired statistics.

Never opens outcomes before all 768 result/seal pairs exist with zero aborts.
No physics, fitting, optional stopping or seed selection occurs here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from prove.fast_adapter.p61_data import sha256
from tools.fast_adapter.p66_report import (ARMS, SPLITS, COMPARISONS, PHASES, need, read, ref,
    key, same, audit_prefix, audit_trace, audit_lifecycle, replay_neural_commands)

from tools.fast_adapter.p67_common import cells, FREEZE_SCHEMA, SEED_START, CONDITIONS_PER_SPLIT

SEEDS={s:tuple(range(SEED_START[s],SEED_START[s]+CONDITIONS_PER_SPLIT)) for s in SPLITS}
BOOTSTRAP_SEED=670067
BOOTSTRAP_REPLICATES=10000



def status(directory):
    root=Path(directory)
    results={p.parent for p in root.glob('*/RESULT.json')}
    seals={p.parent for p in root.glob('*/SEAL.json')}
    aborts=list(root.glob('*/ABORT.json'))+list(root.glob('*/PROGRAM_ERROR.json'))
    return dict(expected=768,result_files=len(results),seal_files=len(seals),aborts=len(aborts),
        complete=len(results)==768 and results==seals and not aborts)


def wilson(k,n):
    need(n>0 and 0<=k<=n,'Invalid binomial counts')
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def exact_mcnemar(rescues,harms):
    need(rescues>=0 and harms>=0,'Invalid discordant counts')
    n=rescues+harms
    return 1. if n==0 else float(binomtest(rescues,n,.5,alternative='two-sided').pvalue)


def holm(pvalues):
    values=np.asarray(pvalues,dtype=float)
    need(values.ndim==1 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'Invalid p values')
    order=np.argsort(values,kind='stable');out=np.empty(len(values));running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(values)-rank)*values[index]);out[index]=min(1.,running)
    return out.tolist()


def paired_stats(a,b,indices):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool)
    need(a.shape==b.shape and a.ndim==1 and len(a)>0,'Invalid paired outcomes')
    delta=b.astype(int)-a.astype(int)
    rescues=int((delta==1).sum());harms=int((delta==-1).sum())
    ci=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()
    return dict(n=len(a),rescues=rescues,harms=harms,ties=int((delta==0).sum()),
        both_success=int((a&b).sum()),both_failure=int((~a&~b).sum()),
        delta=float(delta.mean()),delta_exploratory_bootstrap95=ci)


def aggregate(runs):
    lookup={key(r['result']):r for r in runs}
    need(len(runs)==768 and set(lookup)=={key(dict(cell=c)) for c in cells()},'Exact 768-cell denominator required')
    groups={};comparisons={};pairs=[]
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    for split in SPLITS:
        # The same resampled matched conditions serve all six contrasts in a stratum.
        indices=rng.integers(0,64,size=(BOOTSTRAP_REPLICATES,64))
        for arm in ARMS:
            selected=[lookup[split,seed,arm] for seed in SEEDS[split]]
            success=sum(bool(r['result']['validated_success']) for r in selected)
            groups[split+'/'+arm]=dict(n=64,success=success,rate=success/64,wilson95=wilson(success,64),
                sustained_native_success=sum(bool(r['result']['sustained_native_success']) for r in selected),
                mechanical_anomalies=sum(not r['result']['mechanical_valid'] for r in selected),
                phase_entered={p:sum(p in r['phase_first_t'] for r in selected) for p in PHASES},
                terminal_phases=dict(Counter(r['terminal_phase'] for r in selected)),
                did_not_reach_close=sum('close' not in r['phase_first_t'] for r in selected),
                nn_command_steps=sum(r['nn_command_steps'] for r in selected),
                code_command_steps=sum(r['code_command_steps'] for r in selected),
                update_events=sum(r['update_events'] for r in selected),
                phase_steps={p:sum(r['phase_steps'].get(p,0) for r in selected) for p in PHASES},
                entry_events=sum(r['result']['entry_events'] for r in selected),
                continuous_events=sum(r['result']['continuous_events'] for r in selected),
                reference_missing_frames=sum(r['result']['reference_missing_frames'] for r in selected),
                initialized_conditions=sum(r['entry_initialization'] is not None for r in selected))
        for label,(base,candidate,interpretation) in COMPARISONS.items():
            a=[bool(lookup[split,seed,base]['result']['validated_success']) for seed in SEEDS[split]]
            b=[bool(lookup[split,seed,candidate]['result']['validated_success']) for seed in SEEDS[split]]
            stats=paired_stats(a,b,indices);stats.update(baseline=base,candidate=candidate,interpretation=interpretation)
            if label in ('D-A','D-C'):
                stats['exact_mcnemar_two_sided_p']=exact_mcnemar(stats['rescues'],stats['harms'])
                stats['family']='primary_D-A' if label=='D-A' else 'secondary_D-C'
            else:stats['family']='descriptive_only'
            comparisons[split+'/'+label]=stats
            for seed,x,y in zip(SEEDS[split],a,b):
                pairs.append(dict(split=split,seed=seed,comparison=label,baseline_success=x,candidate_success=y,
                    rescue=y and not x,harm=x and not y))
    for label in ('D-A','D-C'):
        corrected=holm([comparisons[s+'/'+label]['exact_mcnemar_two_sided_p'] for s in SPLITS])
        for split,p in zip(SPLITS,corrected):
            comparisons[split+'/'+label].update(holm_three_strata_p=p,holm_reject_05=p<.05,
                positive_improvement_supported=p<.05 and comparisons[split+'/'+label]['delta']>0)
    pooled={arm:dict(n=192,success=sum(groups[s+'/'+arm]['success'] for s in SPLITS),
        macro_rate=float(np.mean([groups[s+'/'+arm]['rate'] for s in SPLITS])),descriptive_only=True) for arm in ARMS}
    return groups,pairs,comparisons,pooled


def audit_condition(runs):
    need(len(runs)==4 and {r['result']['cell']['arm'] for r in runs}==set(ARMS), 'Exactly four condition arms required')
    split=runs[0]['result']['cell']['split'];seed=runs[0]['result']['cell']['seed']
    need(all(r['result']['cell']['split']==split and r['result']['cell']['seed']==seed for r in runs), 'Condition mixed')
    selected = sorted(runs, key=lambda r: ARMS.index(r['result']['cell']['arm']))
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
    return dict(split=split, seed=seed, four_arm_prefix=common,
                         first_approach_t=entry[0], entry_once_continuous_prefix=continued,
                         identical_entry_initialization=True)

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
    need(0 <= pre['index'] < 768 and pre['cell']==result['cell']==freeze['cells'][pre['index']], 'Cell identity differs')
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


def report(freeze_path,directory,output):
    need(status(directory)['complete'],'Require all 768 result/seal pairs and zero aborts before opening outcomes')
    freeze=read(freeze_path)
    need(freeze.get('schema')==FREEZE_SCHEMA and freeze.get('status')=='frozen'
         and freeze.get('steps')==240 and freeze.get('cells')==cells(),'Wrong P67 frozen scope')
    for path,digest in freeze['files'].items():need(sha256(path)==digest,'Frozen file changed: '+path)
    paths={}
    for path in sorted(Path(directory).glob('*/RESULT.json')):
        identity=key(read(path));need(identity not in paths,'Duplicate condition/arm result');paths[identity]=path.parent
    need(set(paths)=={key(dict(cell=c)) for c in cells()},'Missing or extra condition identity')
    light=[];pairing=[];command_cells=[];replay_metadata=None
    for split in SPLITS:
        for seed in SEEDS[split]:
            runs=[load_cell(paths[split,seed,arm],freeze,freeze_path) for arm in ARMS]
            pairing.append(audit_condition(runs))
            replay=replay_neural_commands(runs,freeze['action_policy'])
            command_cells.extend(replay['cells'])
            replay_metadata={k:v for k,v in replay.items() if k!='cells'}
            light.extend({k:v for k,v in r.items() if k not in ('trace','observations')} for r in runs)
            del runs,replay
            gc.collect()
    groups,pairs,comparisons,pooled=aggregate(light)
    value=dict(schema='prove.p67-reference-confirmation-complete-audit.v1',complete=True,
        freeze=ref(freeze_path),report_source=ref(__file__),groups=groups,pairs=pairs,comparisons=comparisons,
        pooled_descriptive=pooled,all_runtime_invariants_passed=True,pairing=pairing,
        independent_neural_command_replay=dict(**replay_metadata,cells=command_cells),cells=light,
        physics_steps=0,optimizer_updates=0,
        statistics=dict(success_interval='95% Wilson score',paired_test='two-sided exact McNemar: binomial discordants p=0.5',
            primary_family='D-A across three strata; Holm correction within these three tests',
            secondary_family='D-C across three strata; separate Holm correction within these three tests',
            other_contrasts='descriptive; no confirmatory p values',
            bootstrap=dict(seed=BOOTSTRAP_SEED,replicates=BOOTSTRAP_REPLICATES,unit='matched condition within stratum',
                interval='exploratory percentile95; exact tests are primary'),
            pooling='192-condition pooled counts descriptive only; never replace stratum results'),
        definitions=dict(scope='fresh matched confirmation; single cup task and frozen checkpoint; no orientation OOD',
            denominator='all 64 predeclared conditions per stratum and arm, including mechanical/no-entry failures',
            attribution='B-A is reference holding, not gradient benefit; no equal-compute optimizer superiority claim',
            memory='audit one condition/four traces, then release; only lightweight receipts retained'))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'RESULTS.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cells',type=Path,required=True)
    p.add_argument('--freeze',type=Path);p.add_argument('--output',type=Path);p.add_argument('--status-only',action='store_true')
    args=p.parse_args()
    if args.status_only:print(json.dumps(status(args.cells)));return
    if args.freeze is None or args.output is None:p.error('--freeze and --output required')
    value=report(args.freeze,args.cells,args.output);print(json.dumps(value['comparisons'],indent=2))


if __name__=='__main__':main()
