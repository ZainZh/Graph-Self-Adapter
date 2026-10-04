"""Completed P65 only: feedback admission, remaining gates, and ID harms.

Read-only recorded evidence. No model fitting, new inference policy, physics,
or success counterfactual. Precise contour rejection reasons are reconstructed
only at the actually saved public-mask timestamps, never filled in between.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prove.fast_adapter.p60_contour import outer_hull_boundary
from prove.fast_adapter.p61_data import sha256
from prove.fast_adapter.p64_cup import configuration,NOMINAL_ROTATION


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else None


def geometry_record(row,cfg):
    record=row['adaptation'];g=np.asarray(record['geometry8']);z=np.asarray(record['z_after']);T=np.asarray(record['T_world_anchor'])
    center0=T[:3,:3]@g[:3]+T[:3,3];center=T[:3,:3]@(g[:3]+.02*z[:3])+T[:3,3]
    size0=np.exp(g[6:]);size=np.exp(g[6:]+.25*z[6:])
    q=np.asarray(row['state']['qpos'])
    if len(q)!=16 or abs(np.linalg.norm(q[3:7])-1)>1e-5:raise ValueError('Unexpected sealed freejoint layout')
    object_R=Rotation.from_quat(np.r_[q[4:7],q[3]]).as_matrix()
    truth=q[:3]+object_R@np.array([cfg['radius']+cfg['handle_width']/2,-.004,cfg['handle_z']])
    truth_size=np.array([cfg['handle_width']/2+.003,cfg['handle_height']/2+.003])
    route=row['neural_action']
    return dict(t=row['t'],phase_before=row['phase_before'],phase_after=row['phase_after'],U=record['U'],admitted=record['admitted'],
        center_world=center.tolist(),raw_center_world=center0.tolist(),halfsize=size.tolist(),raw_halfsize=size0.tolist(),
        z=z.tolist(),code_target=np.asarray(route['code_target']).tolist(),command_p=np.asarray(row['command_p']).tolist(),
        truth_postseal=dict(center=truth.tolist(),halfsize=truth_size.tolist(),
            center0_error_mm=float(np.linalg.norm(center0-truth)*1000),center_error_mm=float(np.linalg.norm(center-truth)*1000),
            halfsize0_error_mm=float(np.linalg.norm(size0-truth_size)*1000),halfsize_error_mm=float(np.linalg.norm(size-truth_size)*1000)))


def delta(before,after):
    if before is None or after is None:return None
    return dict(from_t=before['t'],to_t=after['t'],
        center_change_mm=float(np.linalg.norm(np.asarray(after['center_world'])-before['center_world'])*1000),
        raw_center_change_mm=float(np.linalg.norm(np.asarray(after['raw_center_world'])-before['raw_center_world'])*1000),
        halfsize_change_mm=float(np.linalg.norm(np.asarray(after['halfsize'])-before['halfsize'])*1000),
        command_change_mm=float(np.linalg.norm(np.asarray(after['command_p'])-before['command_p'])*1000),
        code_target_change_mm=float(np.linalg.norm(np.asarray(after['code_target'])-before['code_target'])*1000),
        latent_change=float(np.linalg.norm(np.asarray(after['z'])-before['z'])))


def run(report,output):
    report,output=Path(report),Path(output)
    if output.exists():raise FileExistsError(output)
    summary=json.loads(report.read_text())
    if not summary.get('complete') or len(summary['cells'])!=36 or not summary['all_runtime_invariants_passed']:
        raise ValueError('Completed audited36 required')
    records=[];lookup={}
    for entry in summary['cells']:
        path=Path(entry['path']);seal=json.loads((path/'SEAL.json').read_text())
        for name,digest in seal['files'].items():
            if sha256(path/name)!=digest:raise ValueError('Changed sealed evidence '+str(path/name))
        trace=np.load(path/'TRACE.npy',allow_pickle=True).tolist();obs=np.load(path/'OBS.npy',allow_pickle=True).tolist()
        result=entry['result'];cell=result['cell'];cfg=configuration('cup',cell['seed'],cell['split'])
        if not np.allclose(np.asarray(trace[0]['state']['qpos'])[:2],cfg['center'][:2],atol=1e-5):raise ValueError('Unverified cup state identity')
        fresh=[r for r in trace if r.get('adaptation') is not None and r['adaptation']['t']==r['t']]
        approach=[r for r in fresh if r['phase_before']=='approach'];start=approach[0]['t'] if approach else None
        pre=[r for r in fresh if start is not None and r['t']<start]
        updates=[r for r in fresh if r['adaptation']['U']]
        counts=Counter();perstep=[]
        for row in approach:
            a=row['adaptation'];enabled=a['stage_schedule']['update_enabled']
            if not enabled:reason='arm_fixed_updates_disabled'
            elif not a['admitted']:reason='current_closed_contour_not_admitted'
            elif a['S']<3:reason='insufficient_causal_support'
            else:reason='eligible_current_contour_and_support'
            expected=3 if reason=='eligible_current_contour_and_support' else 0
            if a['U']!=expected:raise ValueError('Feedback scheduling reconstruction differs')
            counts[reason]+=1
            perstep.append(dict(t=row['t'],admitted=a['admitted'],role_valid=a['role_valid'],S=a['S'],U=a['U'],reason=reason,
                latest_support_t=max(a['support_times'],default=None)))
        mask_checks=[]
        with np.load(path/'PUBLIC_IMAGES.npz') as archive:
            for index,t in enumerate(archive['t']):
                row=trace[int(t)]
                if row['phase_before']!='approach':continue
                mask=np.unpackbits(archive['masks_packed'][index,2],axis=-1)[...,:448]
                boundary,info=outer_hull_boundary(mask)
                if (boundary is not None)!=row['adaptation']['admitted']:raise ValueError('Mask admission replay mismatch')
                mask_checks.append(dict(t=int(t),admitted=boundary is not None,reason=info['reason'],opening_visible=info['opening_visible'],
                    visible_goal_pixels=int(mask.sum())))
        gate=[]
        for row in approach[-30:]:
            route=row['neural_action'];target=np.asarray(route['code_target']);actual=np.asarray(obs[row['t']]['position']);command=np.asarray(row['command_p'])
            angle=float(np.linalg.norm(Rotation.from_matrix(np.asarray(route['code_rotation'])@np.asarray(obs[row['t']]['rotation']).T).as_rotvec()))
            gate.append(dict(t=row['t'],neural_to_code_gate_mm=float(np.linalg.norm(command-target)*1000),
                eef_to_neural_mm=float(np.linalg.norm(actual-command)*1000),eef_to_code_gate_mm=float(np.linalg.norm(actual-target)*1000),
                position_gate_pass=bool(np.linalg.norm(actual-target)<.003),angle_rad=angle,angle_gate_pass=angle<.06))
        before_close=[r for r in fresh if r['phase_before'] not in ('close','transport','hold')]
        selected=dict(first_approach=geometry_record(approach[0],cfg) if approach else None,
            last_actual_update=geometry_record(updates[-1],cfg) if updates else None,
            last_preclose_or_budget=geometry_record(before_close[-1],cfg) if before_close else None)
        record=dict(cell=cell,outcome_class=result['outcome_class'],validated_success=result['validated_success'],
            mechanical_valid=result['mechanical_valid'],maximum_penetration_mm=result['max_full_trajectory_contact_penetration']*1000,
            final_phase=result['final_phase'],first_approach_t=start,approach_steps=len(approach),
            preapproach_admitted=sum(r['adaptation']['admitted'] for r in pre),
            last_preapproach_admitted_t=max((r['t'] for r in pre if r['adaptation']['admitted']),default=None),
            approach_admitted=sum(r['adaptation']['admitted'] for r in approach),
            approach_visible_goal=sum(bool(r['adaptation']['role_valid'][2]) for r in approach),
            approach_update_events=sum(bool(r['adaptation']['U']) for r in approach),
            total_update_events=len(updates),update_t=[r['t'] for r in updates],
            approach_reasons=dict(counts),approach_feedback_history=perstep,
            saved_mask_rejection_replay=mask_checks,
            sampled_mask_reason_counts=dict(Counter(r['reason'] for r in mask_checks)),
            tail_approach_gate=dict(n=len(gate),neural_to_code_gate_mm=stats([g['neural_to_code_gate_mm'] for g in gate]),
                eef_to_neural_mm=stats([g['eef_to_neural_mm'] for g in gate]),eef_to_code_gate_mm=stats([g['eef_to_code_gate_mm'] for g in gate]),
                position_gate_pass=sum(g['position_gate_pass'] for g in gate),angle_gate_pass=sum(g['angle_gate_pass'] for g in gate),rows=gate),
            selected_geometry=selected,
            last_update_to_final_change=delta(selected['last_actual_update'],selected['last_preclose_or_budget']),
            approach_entry_to_final_change=delta(selected['first_approach'],selected['last_preclose_or_budget']),
            seal_sha256=sha256(path/'SEAL.json'))
        records.append(record);lookup[cell['split'],cell['seed'],cell['arm']]=record
    harms=[]
    for seed in range(640100,640104):
        fixed,local=lookup['id',seed,'fixed'],lookup['id',seed,'online_local']
        if fixed['validated_success'] and not local['validated_success']:
            harms.append(dict(seed=seed,fixed=fixed['selected_geometry'],local=local['selected_geometry'],
                local_last_update_to_final_change=local['last_update_to_final_change'],local_terminal_phase=local['final_phase'],
                fixed_outcome=fixed['outcome_class'],local_outcome=local['outcome_class']))
    value=dict(schema='prove.p65-postseal-feedback-and-gate-diagnostic.v1',report_sha256=sha256(report),source_sha256=sha256(__file__),
        physics_steps=0,training_updates=0,runtime_changes=0,
        limits=['Exact rejection reasons reconstructed only for saved masks every4steps; other frames retain recorded admitted=false without inventing a subtype.',
            'Geometry truth reconstructed only in postseal analysis from frozen condition and sealed freejoint pose; never supplied to runtime.',
            'Same-frame geometry and gate differences are observations, not new physical success counterfactuals or a unique causal explanation.',
            'Original mechanical-anomaly outcomes and all36 denominators preserved.'],
        id_local_harms=harms,cells=records)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.report,a.output)
