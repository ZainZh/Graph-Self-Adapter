#!/usr/bin/env python3
"""P62 public-first bounded update-block comparison on opened development data.

No network training, evaluator labels in adaptation, or physical rollout.
Exactly 12 conditions x 2 frozen checkpoints x 5 preregistered update arms are
sealed before any private scoring. P61 sources and artifacts remain immutable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from prove.fast_adapter.p60_contour import model_state_sha256
from prove.fast_adapter.p61_adaptation import adapt_z as legacy_adapt_z
from prove.task_graph.p59.core import FrozenP50Action
from tools.fast_adapter.p61_train import contour_measurements, encode_frames
from tools.fast_adapter.p61_evaluate import (
    MODEL_ARMS, FRAME_INDICES, SUPPORT_TIMES, METRIC_KEYS, DEFAULT_TRAINING_FREEZE,
    _dev_records, _verify_training_freeze, _load_final_checkpoint, _world_opening,
    _metrics, _finite_row, binding, sha256, write_new,
)

BASELINE = Path('evaluations/fast_adapter_p61_task_visual_v01/offline_v01')

ARMS = ('fixed', 'legacy_full8', 'translation3', 'translation_size5', 'full8_spectral')
CANDIDATES = ('full8_spectral', 'translation3', 'translation_size5')
SOURCE_CLOSURE = (
    'tools/fast_adapter/p62_evaluate.py', 'prove/fast_adapter/p62_adaptation.py',
    'tools/fast_adapter/p61_evaluate.py', 'tools/fast_adapter/p61_train.py',
    'prove/fast_adapter/p61_encoder.py', 'prove/fast_adapter/p61_adaptation.py',
    'prove/fast_adapter/p60_contour.py', 'prove/task_graph/p59/core.py',
)


def verify_freeze(path):
    value = json.loads(Path(path).read_text())
    for source, digest in value['source_bindings'].items():
        if sha256(source) != digest:
            raise ValueError(f'P62 frozen source changed: {source}')
    if not set(SOURCE_CLOSURE).issubset(value['source_bindings']):
        raise ValueError('P62 freeze lacks evaluator source closure')
    if tuple(value['adaptation_arms']) != ARMS:
        raise ValueError('P62 preregistered candidate set changed')
    for name in ('PUBLIC_PREDICTIONS.json', 'REPORT.json', 'PREDICTION_SEAL.json'):
        source = str(BASELINE / name)
        if value['source_bindings'].get(source) != sha256(source):
            raise ValueError('P62 must freeze immutable P61 reference artifacts')
    return value


def predict_episode(model, public, device):
    from prove.fast_adapter.p62_adaptation import adapt_z as bounded_adapt_z
    frames = tuple(int(t) for t in public['frame_index'])
    if frames != FRAME_INDICES:
        raise ValueError('P62 frame schedule changed')
    lookup = {t:i for i,t in enumerate(frames)}
    if not all(bool(public['boundary_valid'][lookup[t]]) for t in SUPPORT_TIMES):
        raise ValueError('P62 required public contour missing')
    with torch.no_grad():
        geometry = encode_frames(model, public, [lookup[t] for t in (16,24,29)], device)
    if geometry.shape != (3,8) or not torch.isfinite(geometry).all():
        raise ValueError('invalid P62 frozen encoding')
    transform = public['T_world_anchor'].to(device).float()
    arms = {}
    for arm in ARMS:
        z = torch.zeros(8,device=device)
        updates = []
        if arm != 'fixed':
            for index, times in enumerate(((0,8,16), SUPPORT_TIMES)):
                measurements = contour_measurements(public,[lookup[t] for t in times])
                with torch.enable_grad():
                    if arm == 'legacy_full8':
                        z, records = legacy_adapt_z(geometry[index],z,transform,measurements,create_graph=False)
                    else:
                        z, records = bounded_adapt_z(geometry[index],z,transform,measurements,
                                                     candidate=arm,create_graph=False)
                if len(records)!=3 or not torch.isfinite(z).all():
                    raise FloatingPointError('P62 adaptation incomplete or nonfinite')
                updates.append({'at_t':times[-1],'support_times':list(times),
                                'steps':records,'z_after':z.detach().cpu().tolist()})
        arms[arm] = {'z':z.detach().cpu().tolist(),'updates':updates,
                     **_world_opening(geometry[2],z,transform)}
    return {'query_t':29,'query_measurement_excluded':True,
            'query_freshly_encoded':True,'support_measurement_times':list(SUPPORT_TIMES),
            'query_geometry8':geometry[2].detach().cpu().tolist(),'arms':arms}


def predict(dataset, frozen_checkpoint, task_checkpoint, output, device, freeze):
    verify_freeze(freeze)
    _verify_training_freeze(DEFAULT_TRAINING_FREEZE,dataset)
    info, records = _dev_records(dataset)
    output.mkdir(parents=True,exist_ok=False)
    checkpoints = dict(zip(MODEL_ARMS,(frozen_checkpoint,task_checkpoint)))
    episodes = [{'seed':int(r['seed']),'distribution':r['distribution'],
                 'public':binding(Path(r['path'])/'public.pt'),'models':{}} for r in records]
    metadata = {}
    from prove.fast_adapter.p62_adaptation import candidate_configuration
    candidate_configs = {arm: candidate_configuration(arm) for arm in ARMS}
    for arm in MODEL_ARMS:
        model, meta, checkpoint_hash = _load_final_checkpoint(checkpoints[arm],arm,device)
        before = model_state_sha256(model)
        for record, episode in zip(records,episodes):
            public = torch.load(Path(record['path'])/'public.pt',weights_only=True,map_location='cpu')
            episode['models'][arm] = predict_episode(model,public,device)
        after = model_state_sha256(model)
        if before != after or sha256(checkpoints[arm]) != checkpoint_hash:
            raise RuntimeError('P62 frozen model changed')
        metadata[arm] = dict(meta,model_state_sha256_after=after)
        del model
    baseline_match = compare_baseline_predictions(episodes)
    value = {'baseline_prediction_match': baseline_match, 'schema':'prove.p62-public-dev-predictions.v1','freeze':binding(freeze),
             'dataset':info,'models':metadata,'episodes':episodes,'candidate_configuration':candidate_configs,
             'arms':[f'{m}/{a}' for m in MODEL_ARMS for a in ARMS],
             'private_truth_opened':False,'development_data_opened_previously':True,
             'physical_success_claim':False,'physics_integration_steps':0}
    write_new(output/'PUBLIC_PREDICTIONS.json',value)
    write_new(output/'PREDICTION_SEAL.json',{
        'schema':'prove.p62-public-dev-prediction-seal.v1','freeze':binding(freeze),
        'predictions':binding(output/'PUBLIC_PREDICTIONS.json'),'episodes':12,'prediction_arms':120,
        'dataset_manifest':info['manifest'],'checkpoints':{a:binding(p) for a,p in checkpoints.items()},
        'source_closure':{n:binding(n) for n in SOURCE_CLOSURE},
        'private_truth_opened':False,'sealed_unix':time.time()})
    return value


def compare_baseline_predictions(episodes):
    reference=json.loads((BASELINE/'PUBLIC_PREDICTIONS.json').read_text())
    lookup={(int(r['seed']),r['distribution']):r for r in reference['episodes']}
    maximum=0.
    for row in episodes:
        old=lookup[(row['seed'],row['distribution'])]
        for model in MODEL_ARMS:
            for newarm,oldarm in (('fixed','fixed'),('legacy_full8','online')):
                for key in ('z','center_world','rotation_world','opening_half_size'):
                    new=np.asarray(row['models'][model]['arms'][newarm][key])
                    previous=np.asarray(old['models'][model][oldarm][key])
                    maximum=max(maximum,float(np.max(np.abs(new-previous))))
    if maximum>1e-6:
        raise RuntimeError(f'P61 baseline prediction mismatch is engineering failure: {maximum}')
    return {'passed':True,'maximum_absolute_difference':maximum,'tolerance':1e-6,
            'reference':binding(BASELINE/'PUBLIC_PREDICTIONS.json')}


def compare_baseline_metrics(rows):
    reference=json.loads((BASELINE/'REPORT.json').read_text())
    lookup={(int(r['seed']),r['distribution']):r for r in reference['rows']}
    maximum=0.
    for row in rows:
        old=lookup[(row['seed'],row['distribution'])]
        for model in MODEL_ARMS:
            for newarm,oldarm in (('fixed','fixed'),('legacy_full8','online')):
                for key in METRIC_KEYS:
                    maximum=max(maximum,abs(row['arms'][f'{model}/{newarm}'][key]-old['arms'][f'{model}/{oldarm}'][key]))
    if maximum>1e-4:
        raise RuntimeError(f'P61 baseline metric mismatch is engineering failure: {maximum}')
    return {'passed':True,'maximum_absolute_difference':maximum,'tolerance':1e-4,
            'reference':binding(BASELINE/'REPORT.json')}


def aggregate(rows):
    groups, comparisons = [], []
    for model in MODEL_ARMS:
        for distribution in ('id','geometry'):
            selected = [r for r in rows if r['distribution']==distribution]
            if len(selected)!=6:
                raise ValueError('P62 stratum incomplete')
            for arm in ARMS:
                groups.append({'model_arm':model,'adaptation':arm,'distribution':distribution,'n':6,
                    'metrics':{k:{'mean':float(np.mean([r['arms'][f'{model}/{arm}'][k] for r in selected])),
                                  'median':float(np.median([r['arms'][f'{model}/{arm}'][k] for r in selected]))}
                               for k in METRIC_KEYS}})
                if arm=='fixed':continue
                metrics={}
                for key in METRIC_KEYS:
                    candidate=np.array([r['arms'][f'{model}/{arm}'][key] for r in selected])
                    fixed=np.array([r['arms'][f'{model}/fixed'][key] for r in selected])
                    delta=candidate-fixed
                    metrics[key]={'wins':int((delta < -1e-9).sum()),'losses':int((delta > 1e-9).sum()),
                        'ties':int((np.abs(delta)<=1e-9).sum()),'mean_candidate':float(candidate.mean()),
                        'mean_fixed':float(fixed.mean()),'mean_delta':float(delta.mean()),
                        'max_regression':float(delta.max()),
                        'relative_mean_improvement':float(1-candidate.mean()/max(fixed.mean(),1e-12))}
                comparisons.append({'model_arm':model,'adaptation':arm,'distribution':distribution,'metrics':metrics})
    return {'groups':groups,'paired_comparisons':comparisons}


def qualify(summary):
    checks=[]
    for arm in CANDIDATES:
        strata=[]
        for distribution in ('id','geometry'):
            item=next(r for r in summary['paired_comparisons'] if r['model_arm']=='task_trained_lila'
                      and r['adaptation']==arm and r['distribution']==distribution)
            position=item['metrics']['world_action_position_error_mm_all4phases']
            rotation=item['metrics']['rotation_error_deg']
            conditions={'action_position_mean_improves_at_least20percent':position['relative_mean_improvement']>=.20,
                        'action_position_at_least4of6wins':position['wins']>=4,
                        'geometry_rotation_mean_regression_at_most1deg':rotation['mean_delta']<=1.,
                        'geometry_rotation_any_regression_at_most3deg':rotation['max_regression']<=3.}
            strata.append({'distribution':distribution,'checks':conditions,'passed':all(conditions.values())})
        checks.append({'candidate':arm,'strata':strata,'passed':all(s['passed'] for s in strata)})
    eligible=[r['candidate'] for r in checks if r['passed']]
    return {'primary_model':'task_trained_lila','replication_model':'frozen_lila',
            'candidate_checks':checks,'preference_order':list(CANDIDATES),
            'selected_candidate':eligible[0] if eligible else None,
            'passed':bool(eligible),'new_physical_execution_authorized':False}


def score(output,dataset,p50_checkpoint):
    seal=json.loads((output/'PREDICTION_SEAL.json').read_text())
    if (seal.get('schema')!='prove.p62-public-dev-prediction-seal.v1' or seal.get('episodes')!=12
        or seal.get('prediction_arms')!=120 or seal.get('private_truth_opened') is not False):
        raise ValueError('P62 complete prediction seal missing')
    if set(seal['source_closure'])!=set(SOURCE_CLOSURE):raise ValueError('source closure incomplete')
    for item in [seal['freeze'],seal['predictions'],seal['dataset_manifest'],
                 *seal['checkpoints'].values(),*seal['source_closure'].values()]:
        if sha256(item['path'])!=item['sha256']:raise ValueError('sealed artifact changed')
    verify_freeze(seal['freeze']['path'])
    if sha256(dataset/'MANIFEST.json')!=seal['dataset_manifest']['sha256']:raise ValueError('dataset changed')
    predictions=json.loads((output/'PUBLIC_PREDICTIONS.json').read_text())
    if predictions.get('private_truth_opened') is not False:raise ValueError('public-first boundary invalid')
    _,records=_dev_records(dataset)
    actual={(int(r['seed']),r['distribution']):r for r in predictions['episodes']}
    expected={(int(r['seed']),r['distribution']) for r in records}
    if len(predictions['episodes'])!=12 or set(actual)!=expected:raise ValueError('condition denominator drift')
    # Validate every public prediction arm before opening even the first label.
    for episode in actual.values():
        if set(episode['models']) != set(MODEL_ARMS):
            raise ValueError('model denominator drift')
        for pred in episode['models'].values():
            if (set(pred['arms']) != set(ARMS) or pred['query_t'] != 29
                or pred['query_measurement_excluded'] is not True
                or pred['query_freshly_encoded'] is not True
                or pred['support_measurement_times'] != list(SUPPORT_TIMES)):
                raise ValueError('candidate/query contract drift')
            _finite_row(pred)
    p50=FrozenP50Action(p50_checkpoint).eval().requires_grad_(False)
    before=model_state_sha256(p50);rows=[]
    for record in records:
        identity=int(record['seed']),record['distribution'];episode=actual[identity]
        if set(episode['models'])!=set(MODEL_ARMS):raise ValueError('model denominator drift')
        for model in MODEL_ARMS:
            pred=episode['models'][model]
            if (set(pred['arms'])!=set(ARMS) or pred['query_t']!=29 or not pred['query_measurement_excluded']
                or not pred['query_freshly_encoded'] or pred['support_measurement_times']!=list(SUPPORT_TIMES)):
                raise ValueError('candidate/query contract drift')
        private_path=Path(record['path'])/'private.pt'
        if sha256(private_path)!=record['private_sha256']:raise ValueError('private binding changed')
        public=torch.load(Path(record['path'])/'public.pt',weights_only=True,map_location='cpu')
        private=torch.load(private_path,weights_only=True,map_location='cpu')
        index=public['frame_index'].tolist().index(29)
        row={'seed':identity[0],'distribution':identity[1],'private':binding(private_path),'arms':{}}
        for model in MODEL_ARMS:
            for arm in ARMS:
                row['arms'][f'{model}/{arm}']=_metrics(episode['models'][model]['arms'][arm],
                    private['geometry12_world'][index].float(),public['q'][index].float(),p50)
        _finite_row(row);rows.append(row)
    if before!=model_state_sha256(p50):raise RuntimeError('P50 changed')
    baseline_match=compare_baseline_metrics(rows)
    summary=aggregate(rows);qualification=qualify(summary)
    report={'schema':'prove.p62-private-postseal-dev-report.v1','prediction_seal':binding(output/'PREDICTION_SEAL.json'),
            'baseline_metric_match':baseline_match,'frozen_p50':binding(p50_checkpoint),'rows':rows,'aggregate':summary,'qualification':qualification,
            'truth_opened_only_after_complete_prediction_seal':True,'development_data_opened_previously':True,
            'independent_test_claim':False,'physical_success_claim':False,'physics_integration_steps':0,
            'held_length_use':'evaluator-only true length shared by prediction and teacher'}
    write_new(output/'REPORT.json',report);write_new(output/'QUALIFICATION.json',qualification)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('predict','score'))
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--frozen-checkpoint',type=Path)
    parser.add_argument('--task-checkpoint',type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--p50-checkpoint',type=Path,default=Path('evaluations/fast_adapter_p50_precision_policy_v01/offline_v01/precision_action.pt'))
    args=parser.parse_args();torch.set_num_threads(1)
    if args.operation=='predict':
        result=predict(args.dataset,args.frozen_checkpoint,args.task_checkpoint,args.output,args.device,args.freeze)
        print(json.dumps({'episodes':len(result['episodes']),'arms':result['arms']}))
    else:
        result=score(args.output,args.dataset,args.p50_checkpoint)
        print(json.dumps(result['qualification'],indent=2))


if __name__=='__main__':main()
