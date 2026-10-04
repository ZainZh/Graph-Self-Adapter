"""Completed-only P66 diagnosis; recorded actual decoder base is authoritative.

No policy inference, optimization, physics, or counterfactual success scoring.
Truth is reconstructed exclusively after sealing and remains diagnostic-only.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prove.fast_adapter.p61_data import sha256
from prove.fast_adapter.p64_cup import configuration


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else None


def public_geometry(record):
    """Use stored decoded output; independently check its actual base + z.

    raw_geometry8 is NEVER substituted for decoder_base. The two deliberately
    differ for reference arms and are reported as separate public estimates.
    """
    T=np.asarray(record['T_world_anchor']);raw=np.asarray(record['raw_geometry8'])
    def base_geometry(value):
        if value is None:return None
        value=np.asarray(value)
        return dict(center=(T[:3,:3]@value[:3]+T[:3,3]).tolist(),halfsize=np.exp(value[6:]).tolist())
    decoded=record['decoded'];actual=None;recompute_error=None
    if decoded is not None:
        if record['decoder_base'] is None:raise ValueError('Decoded output lacks actual base')
        base=np.asarray(record['decoder_base']);z=np.asarray(record['z_after'])
        p=T[:3,:3]@(base[:3]+.02*z[:3])+T[:3,3];size=np.exp(base[6:]+.25*z[6:])
        recompute_error=dict(center_m=float(np.linalg.norm(p-decoded[0])),halfsize_m=float(np.linalg.norm(size-decoded[2])))
        if max(recompute_error.values())>1e-6:raise ValueError('Stored decoded output differs from actual base+z')
        actual=dict(center=np.asarray(decoded[0]).tolist(),rotation=np.asarray(decoded[1]).tolist(),halfsize=np.asarray(decoded[2]).tolist())
    return dict(raw=base_geometry(raw),reference=base_geometry(record['g_ref']),actual_decoder_base=base_geometry(record['decoder_base']),
        actual_decoded=actual,decoded_recompute_error=recompute_error,ref_time=record['ref_time'],z=np.asarray(record['z_after']).tolist())


def snapshot(row,cfg):
    a=row['adaptation'];public=public_geometry(a);q=np.asarray(row['state']['qpos'])
    if len(q)!=16 or abs(np.linalg.norm(q[3:7])-1)>1e-5:raise ValueError('Unverified cup freejoint layout')
    rotation=Rotation.from_quat(np.r_[q[4:7],q[3]]).as_matrix()
    truth_center=q[:3]+rotation@np.array([cfg['radius']+cfg['handle_width']/2,-.004,cfg['handle_z']])
    truth_size=np.array([cfg['handle_width']/2+.003,cfg['handle_height']/2+.003])
    errors={}
    for name in ('raw','reference','actual_decoder_base','actual_decoded'):
        value=public[name]
        errors[name]=None if value is None else dict(center_mm=float(np.linalg.norm(np.asarray(value['center'])-truth_center)*1000),
            halfsize_mm=float(np.linalg.norm(np.asarray(value['halfsize'])-truth_size)*1000))
    return dict(t=row['t'],phase_before=row['phase_before'],phase_after=row['phase_after'],event=a['event'],U=a['U'],
        admitted=a['admitted'],public_geometry=public,actual_command=np.asarray(row['command_p']).tolist(),
        code_gate_target=np.asarray(row['neural_action']['code_target']).tolist(),
        postseal_truth_only=dict(center=truth_center.tolist(),halfsize=truth_size.tolist(),errors=errors))


def no_update_reason(a):
    if a['U']:return 'updated_'+a['event']
    if a['arm']=='frame_fixed':return 'frame_fixed_by_design'
    if a['arm']=='reference_fixed':return 'reference_fixed_by_design'
    if not a['reference_available']:return 'no_qualified_reference'
    if a['initialization_pending']:return a['initialization_pending_reason']
    if a['arm']=='entry_once' and a['entry_initialized']:return 'entry_once_completed_by_design'
    if not a['admitted']:return 'no_new_admitted_current_feedback'
    if a['S']<3:return 'fewer_than_three_supports'
    raise ValueError('Unexplained missing P66 update')


def run(report,output):
    report,output=Path(report),Path(output)
    if output.exists():raise FileExistsError(output)
    summary=json.loads(report.read_text())
    if (summary.get('schema')!='prove.p66-reference-state-cup-complete-audit.v1' or not summary.get('complete')
            or len(summary.get('cells',[]))!=48 or not summary.get('all_runtime_invariants_passed')):
        raise ValueError('Completed48 and full runtime audit required before opening cell outcomes')
    records=[]
    for item in summary['cells']:
        path=Path(item['path']);seal_path=path/'SEAL.json';seal=json.loads(seal_path.read_text())
        if sha256(seal_path)!=item['seal_binding']['sha256'] or sha256(path/'RESULT.json')!=item['result_binding']['sha256']:
            raise ValueError('Report binding changed')
        for name,digest in seal['files'].items():
            if sha256(path/name)!=digest:raise ValueError('Changed sealed artifact')
        result=json.loads((path/'RESULT.json').read_text());cell=result['cell'];cfg=configuration('cup',cell['seed'],cell['split'])
        trace=np.load(path/'TRACE.npy',allow_pickle=True).tolist();obs=np.load(path/'OBS.npy',allow_pickle=True).tolist()
        if len(trace)!=240 or len(obs)!=240:raise ValueError('Incomplete H240 trace')
        if not np.allclose(trace[0]['state']['qpos'][:2],cfg['center'][:2],atol=1e-5):raise ValueError('Wrong sealed object layout')
        fresh=[r for r in trace if r.get('adaptation') is not None and r['adaptation']['t']==r['t']]
        approach=[r for r in fresh if r['phase_before']=='approach'];updates=[r for r in fresh if r['adaptation']['U']]
        feedback=[dict(t=r['t'],admitted=r['adaptation']['admitted'],goal_valid=r['adaptation']['goal_valid'],
            S=r['adaptation']['S'],support_times=r['adaptation']['support_times'],U=r['adaptation']['U'],
            event=r['adaptation']['event'],reason=no_update_reason(r['adaptation']),
            reference_available=r['adaptation']['reference_available'],ref_time=r['adaptation']['ref_time'],
            initialization_pending=r['adaptation']['initialization_pending']) for r in approach]
        tail=[]
        for row in approach[-30:]:
            route=row['neural_action'];target=np.asarray(route['code_target']);position=np.asarray(obs[row['t']]['position'])
            angle=float(np.linalg.norm(Rotation.from_matrix(np.asarray(route['code_rotation'])@np.asarray(obs[row['t']]['rotation']).T).as_rotvec()))
            command=np.asarray(row['command_p']);neural=bool(route['used'])
            tail.append(dict(t=row['t'],actual_neural=neural,eef_to_actual_command_mm=float(np.linalg.norm(position-command)*1000),
                eef_to_neural_mm=float(np.linalg.norm(position-command)*1000) if neural else None,
                neural_to_code_gate_mm=float(np.linalg.norm(command-target)*1000) if neural else None,
                eef_to_code_gate_mm=float(np.linalg.norm(position-target)*1000),position_gate_pass=bool(np.linalg.norm(position-target)<.003),
                angle_rad=angle,angle_gate_pass=angle<.06))
        drift=[]
        for first,second in zip(approach,approach[1:]):
            a,b=first['adaptation'],second['adaptation']
            if (a['g_ref'] is not None and b['g_ref'] is not None and np.array_equal(a['g_ref'],b['g_ref'])
                    and np.array_equal(a['z_after'],b['z_after']) and a['decoded'] is not None and b['decoded'] is not None):
                drift.append(dict(from_t=first['t'],to_t=second['t'],center_mm=float(np.linalg.norm(np.asarray(a['decoded'][0])-b['decoded'][0])*1000),
                    halfsize_mm=float(np.linalg.norm(np.asarray(a['decoded'][2])-b['decoded'][2])*1000)))
        for row in fresh:public_geometry(row['adaptation'])
        records.append(dict(cell=cell,outcome_class=result['outcome_class'],validated_success=result['validated_success'],
            mechanical_valid=result['mechanical_valid'],maximum_penetration_m=result['max_full_trajectory_contact_penetration'],final_phase=result['final_phase'],
            entry_times=[r['t'] for r in updates if r['adaptation']['event']=='entry'],
            continuous_times=[r['t'] for r in updates if r['adaptation']['event']=='continuous'],
            entry_events=sum(r['adaptation']['event']=='entry' for r in updates),continuous_events=sum(r['adaptation']['event']=='continuous' for r in updates),
            approach_feedback=feedback,approach_admitted=sum(r['admitted'] for r in feedback),
            no_update_reason_counts=dict(Counter(r['reason'] for r in feedback)),
            tail_approach=dict(n=len(tail),rows=tail,neural_rows=sum(r['actual_neural'] for r in tail),
                eef_to_neural_mm=stats([r['eef_to_neural_mm'] for r in tail if r['actual_neural']]),
                neural_to_code_gate_mm=stats([r['neural_to_code_gate_mm'] for r in tail if r['actual_neural']]),
                eef_to_code_gate_mm=stats([r['eef_to_code_gate_mm'] for r in tail]),
                position_gate_pass=sum(r['position_gate_pass'] for r in tail),angle_gate_pass=sum(r['angle_gate_pass'] for r in tail)),
            first_approach=snapshot(approach[0],cfg) if approach else None,last_update=snapshot(updates[-1],cfg) if updates else None,
            last_preclose_or_budget=snapshot(fresh[-1],cfg) if fresh else None,
            unchanged_reference_z_drift=dict(n=len(drift),center_mm=stats([r['center_mm'] for r in drift]),
                halfsize_mm=stats([r['halfsize_mm'] for r in drift]),rows=drift),
            seal_binding=dict(path=str(seal_path),sha256=sha256(seal_path)),result_binding=item['result_binding']))
    result=dict(schema='prove.p66-completed-postseal-diagnostic.v1',report_binding=dict(path=str(report),sha256=sha256(report)),
        source_binding=dict(path=__file__,sha256=sha256(__file__)),cells=records,n=48,physics_steps=0,training_updates=0,policy_inference_calls=0,
        limits=['Actual decoded state comes from recorded decoder_base/g_ref+z, not moving raw_geometry8.',
            'Truth reconstructed only afterseal from fixed configuration and recorded cupfreejoint; never runtime input.',
            'Gate/geometry/drift observations do not identify a unique cause or constitute physical counterfactual success.',
            'Original mechanical anomalies and all48 denominator retained.'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.report,a.output)
