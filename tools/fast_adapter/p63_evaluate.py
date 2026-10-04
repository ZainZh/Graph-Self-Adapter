"""P63 sealed offline 2x2 public-contour comparison; never executes physics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.p60_contour import ContourMeasurement, outer_hull_boundary, model_state_sha256
from prove.fast_adapter.p62_adaptation import adapt_z, candidate_configuration
from prove.fast_adapter.p63_views import TIMES, PUBLIC_ARRAY_KEYS
from prove.task_graph.p59.core import FrozenP50Action, pack_geometry, matrix_to_rotation_6d
from tools.fast_adapter.p61_evaluate import _world_opening, _metrics, _finite_row, METRIC_KEYS, binding, sha256, write_new

ARMS = ('fixed', 'ts5_wrist', 'full8_wrist', 'ts5_dual', 'full8_dual')
SUPPORT = (0, 8, 16, 24)
LATE = (40, 60, 90, 120, 180, 239)
P62_FREEZE = Path('evaluations/fast_adapter_p62_observable_update_v01/physical_v01/PHYSICAL_FREEZE.json')
PLAN = Path('evaluations/fast_adapter_p63_multiview_feedback_v01/PLAN.md')
SOURCE_CLOSURE = ('tools/fast_adapter/p63_evaluate.py', 'prove/fast_adapter/p63_views.py',
 'prove/fast_adapter/p62_adaptation.py', 'prove/fast_adapter/p61_adaptation.py',
 'prove/fast_adapter/p60_contour.py', 'prove/task_graph/p59/core.py',
 'tools/fast_adapter/p61_evaluate.py', 'tools/fast_adapter/p61_train.py',
 'prove/fast_adapter/p61_encoder.py', 'tests/test_p63_evaluate.py')


def checked(item):
    if sha256(item['path']) != item['sha256']:
        raise ValueError('frozen artifact changed: '+item['path'])


def public_records(dataset):
    seal = json.loads((dataset/'PUBLIC_SEAL.json').read_text())
    if seal['states'] != 224 or seal['physics_steps'] != 0:
        raise ValueError('public cache denominator/physics drift')
    if seal['freeze_sha256'] != sha256(dataset/'FREEZE.json'):
        raise ValueError('view freeze drift')
    for name, digest in seal['files'].items():
        if name.startswith('private/') or sha256(dataset/name) != digest:
            raise ValueError('public seal drift')
    folders = sorted((dataset/'public').iterdir())
    records = []
    for folder in folders:
        for name in ('MANIFEST.json', 'VIEWS.npz'):
            if str((folder/name).relative_to(dataset)) not in seal['files']:
                raise ValueError('unsealed public record')
        manifest = json.loads((folder/'MANIFEST.json').read_text())
        records.append(dict(name=folder.name, **manifest['cell']))
    expected = {(a,s,seed) for a in ('fixed','translation_size5')
                for s, start in (('id',611100),('geometry',611200)) for seed in range(start,start+4)}
    if len(records)!=16 or {(r['arm'],r['split'],int(r['condition_seed'])) for r in records} != expected:
        raise ValueError('source cohort denominator drift')
    return records


def prepare_freeze(dataset, output, task_checkpoint, p50_checkpoint):
    records = public_records(dataset)
    viewfreeze = json.loads((dataset/'FREEZE.json').read_text())
    origins = {p:d for section in ('sources','code') for p,d in viewfreeze[section].items()}
    for path, digest in origins.items():
        if sha256(path)!=digest: raise ValueError('P62/view origin drift: '+path)
    prior=json.loads(P62_FREEZE.read_text())
    for item in prior['files'].values(): checked(item)
    if (sha256(task_checkpoint)!=prior['files']['task_trained_lila']['sha256']
        or sha256(p50_checkpoint)!=prior['files']['action_head']['sha256']):
        raise ValueError('P63 must reuse the exact P62 model and action head')
    origins.update({item['path']:item['sha256'] for item in prior['files'].values()})
    result = dict(schema='prove.p63-offline-freeze.v1', arms=list(ARMS),
        source_bindings={p:sha256(p) for p in SOURCE_CLOSURE}, plan=binding(PLAN),
        public_seal=binding(dataset/'PUBLIC_SEAL.json'), view_freeze=binding(dataset/'FREEZE.json'),
        origin_bindings=origins, task_checkpoint=binding(task_checkpoint), p50_checkpoint=binding(p50_checkpoint),
        records=records, support_times=list(SUPPORT), update_times=[16,24], query_t=29,
        inner={a:candidate_configuration('translation_size5' if a.startswith('ts5') else 'legacy_full8')
               for a in ARMS if a!='fixed'}, minimum_unique_support_times=3,
        current_view_required=True, equal_weight_per_valid_view=True,
        geometry_source='same-frame frozen P62 uncorrected geometry8; no reencoding',
        primary_source_arm='translation_size5', replication_source_arm='fixed',
        predictions=80, physics_steps=0)
    output.mkdir(parents=True,exist_ok=True)
    write_new(output/'FREEZE.json',result)
    return result


def verify_freeze(path, review=True):
    frozen=json.loads(Path(path).read_text())
    if frozen['arms']!=list(ARMS) or set(frozen['source_bindings'])!=set(SOURCE_CLOSURE):
        raise ValueError('candidate/source closure drift')
    for p,d in {**frozen['source_bindings'],**frozen['origin_bindings']}.items():
        if sha256(p)!=d: raise ValueError('source changed: '+p)
    for key in ('plan','public_seal','view_freeze','task_checkpoint','p50_checkpoint'): checked(frozen[key])
    if review:
        receipt=json.loads((Path(path).parent/'ROOT_REVIEW.json').read_text())
        if receipt.get('approved') is not True or receipt.get('freeze_sha256')!=sha256(path):
            raise ValueError('root exact-freeze review required')
    return frozen


def validate_public(public):
    if set(public)!=set(PUBLIC_ARRAY_KEYS)|{'public_held_length'}:
        raise ValueError('public allowlist violation')
    n=len(TIMES)
    shapes={'rgb':(n,2,448,448,3),'visible_masks':(n,2,3,448,448), 'K':(n,2,3,3),
            'T_world_camera':(n,2,4,4),'T_world_eef':(n,4,4),'t':(n,),
            'simulation_time':(n,),'p62_geometry8':(n,8),'T_world_anchor':(n,4,4),'public_held_length':()}
    for key,shape in shapes.items():
        if public[key].shape!=shape or not np.isfinite(public[key]).all():
            raise ValueError('invalid public array: '+key)
    if tuple(public['t'])!=TIMES: raise ValueError('public time schedule drift')
    if not np.array_equal(public['T_world_anchor'],np.broadcast_to(public['T_world_anchor'][0],(n,4,4))):
        raise ValueError('episode anchor changed')


def measurements(public, times, dual):
    result=[]
    for t in times:
        i=TIMES.index(t)
        for v in range(2 if dual else 1):
            boundary,info=outer_hull_boundary(public['visible_masks'][i,v,2])
            if boundary is None: continue
            camera=public['T_world_camera'][i,v]
            result.append(ContourMeasurement(boundary,np.asarray(info['hull_vertices'],dtype=np.float32),
                public['K'][i,v],camera[:3,3],camera[:3,:3],(448,448),t))
    return result


def predict_episode(public, device='cpu'):
    validate_public(public)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    anchor=tensor(public['T_world_anchor'][0]); arms={}
    for arm in ARMS:
        z=torch.zeros(8,device=device); updates=[]
        if arm!='fixed':
            for t in (16,24):
                history=measurements(public,[s for s in SUPPORT if s<=t],arm.endswith('dual'))
                observed=sorted({m.t for m in history})
                eligible=len(observed)>=3 and t in observed
                steps=[]
                if eligible:
                    z,steps=adapt_z(tensor(public['p62_geometry8'][TIMES.index(t)]),z,anchor,history,
                        candidate='translation_size5' if arm.startswith('ts5') else 'legacy_full8')
                    if len(steps)!=3: raise RuntimeError('incomplete update')
                updates.append(dict(at_t=t,observed_times=observed,measurements=len(history),
                                    admitted=eligible,steps=steps,z_after=z.tolist()))
        arms[arm]=dict(z=z.tolist(),updates=updates,
                      **_world_opening(tensor(public['p62_geometry8'][TIMES.index(29)]),z,anchor))
    coverage=[dict(t=t,wrist=bool(measurements(public,[t],False)),dual=bool(measurements(public,[t],True))) for t in TIMES]
    result=dict(query_t=29,query_measurement_excluded=True,query_geometry8=public['p62_geometry8'][TIMES.index(29)].tolist(),
                arms=arms,coverage=coverage)
    _finite_row(result)
    return result


def predict(dataset, output, device='cpu'):
    frozen=verify_freeze(output/'FREEZE.json')
    records=public_records(dataset)
    if records!=frozen['records'] or binding(dataset/'PUBLIC_SEAL.json')!=frozen['public_seal']:
        raise ValueError('public cache changed')
    episodes=[]
    for record in records:
        path=dataset/'public'/record['name']/'VIEWS.npz'
        with np.load(path,allow_pickle=False) as public: prediction=predict_episode(public,device)
        episodes.append(dict(source=record,public=binding(path),**prediction))
    result=dict(schema='prove.p63-public-predictions.v1',episodes=episodes,private_truth_opened=False,
                neural_reencoding=False,physics_steps=0)
    write_new(output/'PUBLIC_PREDICTIONS.json',result)
    write_new(output/'PREDICTION_SEAL.json',dict(schema='prove.p63-prediction-seal.v1',
        freeze=binding(output/'FREEZE.json'),predictions=binding(output/'PUBLIC_PREDICTIONS.json'),
        episodes=16,predictions_count=80,private_truth_opened=False))
    return result


def aggregate(rows):
    groups=[]; comparisons=[]
    for source in ('translation_size5','fixed'):
        for split in ('id','geometry'):
            selected=[r for r in rows if r['source']['arm']==source and r['source']['split']==split]
            if len(selected)!=4: raise ValueError('stratum denominator drift')
            for arm in ARMS:
                groups.append(dict(source_arm=source,split=split,arm=arm,n=4,
                    metrics={k:float(np.mean([r['arms'][arm][k] for r in selected])) for k in METRIC_KEYS}))
                if arm=='fixed':continue
                metrics={}
                for k in METRIC_KEYS:
                    candidate=np.array([r['arms'][arm][k] for r in selected]); fixed=np.array([r['arms']['fixed'][k] for r in selected])
                    delta=candidate-fixed
                    metrics[k]=dict(wins=int((delta < -1e-9).sum()),losses=int((delta > 1e-9).sum()),
                        mean_candidate=float(candidate.mean()),mean_fixed=float(fixed.mean()),
                        relative_mean_improvement=float(1-candidate.mean()/max(fixed.mean(),1e-12)),max_regression=float(delta.max()))
                comparisons.append(dict(source_arm=source,split=split,arm=arm,metrics=metrics))
    return dict(groups=groups,comparisons=comparisons)


def qualify(summary, episodes):
    checks={}; coverage={}
    for split in ('id','geometry'):
        item=next(r for r in summary['comparisons'] if r['source_arm']=='translation_size5' and r['split']==split and r['arm']=='full8_dual')
        action=item['metrics']['world_action_position_error_mm_all4phases'];rotation=item['metrics']['rotation_error_deg']
        wrist=next(r['metrics']['rotation_error_deg'] for r in summary['groups'] if r['source_arm']=='translation_size5' and r['split']==split and r['arm']=='full8_wrist')
        checks[split]=dict(action20percent=action['relative_mean_improvement']>=.20,action3of4=action['wins']>=3,
            rotation30percent=rotation['relative_mean_improvement']>=.30,rotation3of4=rotation['wins']>=3,
            maximum_rotation_regression3deg=rotation['max_regression']<=3.,rotation_not_worse_than_wrist=rotation['mean_candidate']<=wrist)
        frames=[f for e in episodes if e['source']['arm']=='translation_size5' and e['source']['split']==split for f in e['coverage'] if f['t'] in LATE]
        if len(frames)!=24:raise ValueError('coverage denominator drift')
        coverage[split]={v:float(np.mean([f[v] for f in frames])) for v in ('wrist','dual')}
        checks[split]['dual_coverage50percent']=coverage[split]['dual']>=.5
    gain=float(np.mean([v['dual']-v['wrist'] for v in coverage.values()]))
    passed=all(all(c.values()) for c in checks.values()) and gain>=.25
    return dict(candidate='full8_dual',checks=checks,late_coverage=coverage,late_coverage_gain= gain,
                coverage_gain25pp=gain>=.25,passed=passed,selected_candidate='full8_dual' if passed else None,
                new_physical_execution_authorized=False,requires_A_and_B_qualification=True)


def score(dataset,output):
    seal=json.loads((output/'PREDICTION_SEAL.json').read_text())
    if seal.get('episodes')!=16 or seal.get('predictions_count')!=80 or seal.get('private_truth_opened') is not False:
        raise ValueError('complete 80-prediction seal required')
    checked(seal['freeze']);checked(seal['predictions']); frozen=verify_freeze(seal['freeze']['path'])
    records=public_records(dataset)
    if records!=frozen['records'] or binding(dataset/'PUBLIC_SEAL.json')!=frozen['public_seal']:raise ValueError('cache drift')
    predictions=json.loads((output/'PUBLIC_PREDICTIONS.json').read_text());episodes=predictions['episodes']
    if predictions['private_truth_opened'] is not False or len(episodes)!=16 or [e['source'] for e in episodes]!=records:raise ValueError('prediction denominator drift')
    for episode in episodes:
        checked(episode['public']);_finite_row(episode)
        if set(episode['arms'])!=set(ARMS) or episode['query_t']!=29 or episode['query_measurement_excluded'] is not True:
            raise ValueError('prediction contract drift')
    # No private path is read until every public arm and binding passes above.
    private_seal=json.loads((dataset/'PRIVATE_SEAL.json').read_text())
    if private_seal['public_seal_sha256']!=frozen['public_seal']['sha256']:
        raise ValueError('private/public seal mismatch')
    for name,digest in private_seal['files'].items():
        if sha256(dataset/name)!=digest:raise ValueError('private artifact seal mismatch')
    labels={}
    for record in records:
        name='private/'+record['name']+'/LABELS.json'
        if name not in private_seal['files']:raise ValueError('private label absent from seal')
        private=json.loads((dataset/name).read_text())
        checked(private['source_config'])
        if (private['source_config']['condition_seed']!=record['condition_seed']
            or private['source_config']['sha256']!=frozen['origin_bindings'][str(P62_FREEZE)]
            or private['public_seal_sha256']!=frozen['public_seal']['sha256']):
            raise ValueError('private identity mismatch')
        labels[record['name']]=private
    p50=FrozenP50Action(frozen['p50_checkpoint']['path']).eval().requires_grad_(False)
    before=model_state_sha256(p50);rows=[]
    for episode in episodes:
        record=episode['source'];private_path=dataset/'private'/record['name']/'LABELS.json'
        private=labels[record['name']];tensor=lambda x:torch.as_tensor(x,dtype=torch.float32)
        target=pack_geometry(tensor(private['center_world']),tensor(private['rotation_world']),tensor(private['opening_half']),tensor([private['evaluator_held_length']]))
        with np.load(episode['public']['path'],allow_pickle=False) as public:
            pose=tensor(public['T_world_eef'][TIMES.index(29)])
        q=torch.cat((pose[:3,3],matrix_to_rotation_6d(pose[:3,:3])))
        rows.append(dict(source=record,private=binding(private_path),arms={a:_metrics(episode['arms'][a],target,q,p50) for a in ARMS}))
    if before!=model_state_sha256(p50):raise RuntimeError('P50 changed')
    summary=aggregate(rows);qualification=qualify(summary,episodes)
    result=dict(schema='prove.p63-postseal-report.v1',rows=rows,aggregate=summary,qualification=qualification,
        prediction_seal=binding(output/'PREDICTION_SEAL.json'),private_seal=binding(dataset/'PRIVATE_SEAL.json'),
        truth_opened_only_after_complete_prediction_seal=True,
        held_length_use='evaluator-only true length shared by prediction and teacher',independent_test_claim=False,physics_steps=0)
    _finite_row(result);write_new(output/'REPORT.json',result);write_new(output/'QUALIFICATION.json',qualification)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare-freeze','predict','score'))
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task-checkpoint',type=Path);parser.add_argument('--p50-checkpoint',type=Path)
    parser.add_argument('--device',default='cpu');args=parser.parse_args();torch.set_num_threads(1)
    if args.operation=='prepare-freeze':prepare_freeze(args.dataset,args.output,args.task_checkpoint,args.p50_checkpoint)
    elif args.operation=='predict':predict(args.dataset,args.output,args.device)
    else:score(args.dataset,args.output)

if __name__=='__main__':main()
