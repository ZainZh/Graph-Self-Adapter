"""P66 sparse public replay: seal all48 predictions before invariant summary.

Only the twelve sealed P65 online_local trajectories are inputs. Stored RGB
and masks exist every4steps; their matching raw LiLa readouts are reused, never
new predictions or dense feedback. Postclose raw readouts were not collected
by P65 and are explicitly absent, with the decoded estimate held instead.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.p66_cup import ARMS,ReferenceState
from prove.fast_adapter.p60_contour import outer_hull_boundary,ContourMeasurement,causal_support
from prove.fast_adapter.p61_data import sha256


def validate(path):
    path=Path(path);f=json.loads(path.read_text());r=json.loads((path.parent/'ROOT_REPLAY_REVIEW.json').read_text())
    if (f.get('schema')!='prove.p66-cup-replay-freeze.v1' or f.get('status')!='frozen'
            or r.get('approved') is not True or r.get('reviewer')!='root' or r.get('freeze_sha256')!=sha256(path)
            or f.get('arms')!=list(ARMS) or len(f.get('sources',[]))!=12):raise ValueError('Exact root-reviewed sparse replay freeze required')
    for name,digest in f['files'].items():
        if sha256(name)!=digest:raise ValueError('Frozen replay source changed: '+name)
    return f


def run(freeze,output):
    f=validate(freeze);output=Path(output);output.mkdir(parents=True,exist_ok=False);bindings=[]
    for source in f['sources']:
        path=Path(source['path']);cell=source['cell']
        if cell['arm']!='online_local':raise ValueError('Only P65 online_local records allowed')
        seal=json.loads((path/'SEAL.json').read_text())
        for name,digest in seal['files'].items():
            if sha256(path/name)!=digest:raise ValueError('Altered sealed P65 record')
        traces=np.load(path/'TRACE.npy',allow_pickle=True).tolist();observations=np.load(path/'OBS.npy',allow_pickle=True).tolist()
        with np.load(path/'PUBLIC_IMAGES.npz') as archive:
            if not np.array_equal(archive['t'],np.arange(0,240,4)):raise ValueError('Expected sixty stored sparse observations')
            for arm in ARMS:
                state=ReferenceState(arm);history=[];predictions=[]
                for index,t in enumerate(archive['t']):
                    t=int(t);row=traces[t];phase=row['phase_before'];public=observations[t]
                    if phase in ('close','transport','hold'):
                        if not state.ended:state.finish(t,'source_recorded_critical_exit')
                        predictions.append(dict(t=t,phase_before=phase,active_update_frame=False,raw_available=False,
                            decoded=state.estimate,state_record=None,reason='P65 raw encoder stopped afterclose; cached decoded prediction only'))
                        continue
                    raw=row['adaptation']
                    if raw['t']!=t:raise ValueError('A fresh raw readout is required beforeclose')
                    masks=np.unpackbits(archive['masks_packed'][index],axis=-1)[...,:448].astype(bool)
                    boundary,info=outer_hull_boundary(masks[2]);admitted=boundary is not None
                    if admitted!=raw['admitted']:raise ValueError('Public contour admission differs from sealed record')
                    if admitted:history.append(ContourMeasurement(boundary,np.asarray(info['hull_vertices'],np.float32),
                        public['K'],public['camera_position'],public['camera_rotation'],(448,448),t))
                    estimate=state.step(torch.tensor(raw['geometry8'],dtype=torch.float32),
                        torch.tensor(raw['T_world_anchor'],dtype=torch.float32),causal_support(history),
                        t=t,phase=phase,admitted=admitted,goal_valid=bool(masks[2].any()))
                    predictions.append(dict(t=t,phase_before=phase,active_update_frame=True,raw_available=True,
                        decoded=estimate,state_record=state.records[-1],public_pose={k:v for k,v in public.items() if k!='t'},
                        admission_reason=info['reason']))
                if not state.ended:state.finish(240,'sparse_budget_end')
                destination=output/f"{cell['split']}_{cell['seed']}_{arm}";destination.mkdir()
                np.save(destination/'PREDICTIONS.npy',np.asarray(predictions,dtype=object),allow_pickle=True)
                np.save(destination/'LIFECYCLE.npy',np.asarray(state.exit_records,dtype=object),allow_pickle=True)
                receipt=dict(cell={**cell,'arm':arm},source=str(path),source_seal_sha256=sha256(path/'SEAL.json'),
                    files={name:sha256(destination/name) for name in ('PREDICTIONS.npy','LIFECYCLE.npy')})
                (destination/'SEAL.json').write_text(json.dumps(receipt,indent=2)+'\n')
                bindings.append(dict(path=str(destination),seal_sha256=sha256(destination/'SEAL.json')))
    # No evaluation/summary reads until every arm's prediction bytes are sealed.
    (output/'PREDICTIONS_SEALED.json').write_text(json.dumps(dict(n=48,bindings=bindings,freeze_sha256=sha256(freeze)),indent=2)+'\n')
    summaries=[]
    for binding in bindings:
        path=Path(binding['path']);seal=json.loads((path/'SEAL.json').read_text())
        for name,digest in seal['files'].items():
            if sha256(path/name)!=digest:raise ValueError('Prediction changed afterseal')
        predictions=np.load(path/'PREDICTIONS.npy',allow_pickle=True).tolist();records=[r['state_record'] for r in predictions if r['state_record'] is not None]
        refs=[r['g_ref'] for r in records if r['g_ref'] is not None]
        fixed_ref=all(np.array_equal(ref,refs[0]) for ref in refs)
        causal=all(max(r['support_times'],default=-1)<=r['t'] and (r['ref_time'] is None or r['ref_time']<=r['t']) for r in records)
        finite=all(np.isfinite(r['raw_geometry8']).all() and np.isfinite(r['z_after']).all() and
            (r['decoded'] is None or all(np.isfinite(v).all() for v in r['decoded'])) for r in records)
        stability=[]
        if seal['cell']['arm']!='frame_fixed':
            for a,b in zip(records,records[1:]):
                if a['phase_before']==b['phase_before']=='approach' and a['g_ref'] is not None and b['U']==0 and a['decoded'] is not None and b['decoded'] is not None:
                    stability.append(all(np.array_equal(x,y) for x,y in zip(a['decoded'],b['decoded'])))
        if not finite or not causal or not fixed_ref or not all(stability):raise ValueError('P66 sparse invariant failed')
        summaries.append(dict(cell=seal['cell'],finite=finite,causal=causal,g_ref_fixed=fixed_ref,
            no_update_stability_checks=len(stability),no_update_stable=all(stability),
            entry_events=sum(r['event']=='entry' for r in records),continuous_events=sum(r['event']=='continuous' for r in records),
            entry_with_current_rejected=sum(r['event']=='entry' and not r['admitted'] for r in records),
            reference_unavailable_frames=sum(r['qualification_reason']=='no_admitted_historical_reference_hold' for r in records)))
    result=dict(schema='prove.p66-sparse-replay-summary.v1',n=48,cells=summaries,all_invariants_passed=True,
        physics_steps=0,training_updates=0,truth_inputs=False,feedback_stride=4,
        scope='Sparse zero-physics public observation replay, not task success, independent generalization or full-density reproduction')
    (output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.freeze,a.output)
