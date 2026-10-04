#!/usr/bin/env python3
"""Private post-seal P62 geometry diagnostic; never a runtime input.

All sixteen completed attempts, trajectory seals, result bindings, and frozen
P50 command audits are revalidated before evaluator configuration geometry is
accessed. This describes closed-loop estimates; it never updates a candidate,
changes selection, trains a model, or constructs a simulator.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tools.task_graph.p62_report import build_report
from tools.task_graph.p62_visual_validation import binding
from tools.task_graph.train_precision_action_policy import dump_new

WINDOWS=('t12','t24','t29','last10','postprefix')
METRICS=('center_error_mm','slot_local_lateral_error_mm','slot_local_depth_signed_mm',
         'slot_local_depth_abs_mm','rotation_error_deg','half_size_mean_abs_error_mm')
ROLE_NAMES=('actor_robot','held_object','goal_slot')


def geometry_metrics(estimate,configuration):
    if estimate is None:return None
    center,rotation,size=[np.asarray(x,dtype=float) for x in estimate]
    if center.shape!=(3,) or rotation.shape!=(3,3) or size.shape!=(2,):
        raise ValueError('malformed runtime opening geometry')
    if not all(np.isfinite(x).all() for x in (center,rotation,size)):
        raise ValueError('nonfinite runtime opening geometry')
    truth_center=np.asarray(configuration['center'],dtype=float)
    truth_rotation=Rotation.from_euler('xyz',configuration['angles'],degrees=True).as_matrix()
    truth_size=np.asarray(configuration['half'])+np.asarray(configuration['clearance'])
    delta=center-truth_center;local=truth_rotation.T@delta
    angle=Rotation.from_matrix(truth_rotation.T@rotation).magnitude()
    return {'center_error_mm':float(np.linalg.norm(delta)*1000),
            'center_world_error_xyz_mm':(delta*1000).tolist(),
            'slot_local_error_xyz_mm':(local*1000).tolist(),
            'slot_local_lateral_error_mm':float(np.linalg.norm(local[:2])*1000),
            'slot_local_depth_signed_mm':float(local[2]*1000),
            'slot_local_depth_abs_mm':float(abs(local[2])*1000),
            'rotation_error_deg':float(np.rad2deg(angle)),
            'half_size_signed_error_xy_mm':((size-truth_size)*1000).tolist(),
            'half_size_mean_abs_error_mm':float(np.abs(size-truth_size).mean()*1000)}


def window_summary(traces,configuration,times):
    selected=[traces[t] for t in times]
    values=[geometry_metrics(row['estimate'],configuration) for row in selected]
    valid=[value for value in values if value is not None]
    return {'times':list(times),'expected_estimates':len(selected),'valid_estimates':len(valid),
            'missing_estimates':len(selected)-len(valid),
            'mean':{key:float(np.mean([r[key] for r in valid])) if valid else None for key in METRICS},
            'individual':values if len(times)==1 else None}


def diagnostic(acceptance_path):
    acceptance_path=Path(acceptance_path)
    accepted=json.loads(acceptance_path.read_text())
    if (accepted.get('schema')!='prove.p62-physical-acceptance-report.v1'
        or accepted.get('expected_cells')!=16 or accepted.get('completed')!=16
        or accepted.get('aborted')!=0 or accepted.get('all_attempts_ended') is not True
        or accepted.get('all_cells_engineering_valid') is not True):
        raise ValueError('diagnostic requires all sixteen completed and audited physical attempts')
    freeze_path=Path(accepted['freeze']['path'])
    if binding(freeze_path)!=accepted['freeze']:raise ValueError('accepted freeze changed')
    # Reuses the seal and P50/latent audit, without constructing the environment.
    revalidated=build_report(freeze_path)
    if revalidated['completed']!=16 or not revalidated['all_cells_engineering_valid']:
        raise ValueError('physical artifacts no longer pass acceptance')
    accepted_rows={row['directory']:row for row in accepted['rows']}
    if len(accepted_rows)!=16:raise ValueError('acceptance denominator drift')
    for row in revalidated['rows']:
        old=accepted_rows.get(row['directory'])
        if old is None or any(old[key]!=row[key] for key in ('cell','result_binding','seal','pre_run')):
            raise ValueError('accepted result/seal binding changed')
    # Private geometry is accessed only after the complete cohort admission above.
    frozen=json.loads(freeze_path.read_text())
    configurations={(r['split'],int(r['condition_seed'])):r['evaluator_configuration']
                    for r in frozen['conditions']}
    rows=[]
    for accepted_row in revalidated['rows']:
        cell=accepted_row['cell'];directory=Path(accepted_row['directory'])
        cfg=configurations[(cell['split'],int(cell['condition_seed']))]
        traces=np.load(directory/'TRACE.npy',allow_pickle=True).tolist()
        if len(traces)!=240 or [int(r['t']) for r in traces]!=list(range(240)):
            raise ValueError('trace time identity changed')
        schedules={'t12':[12],'t24':[24],'t29':[29],'last10':list(range(230,240)),
                   'postprefix':list(range(12,240))}
        windows={name:window_summary(traces,cfg,times) for name,times in schedules.items()}
        phases=Counter();missing=Counter({role:0 for role in ROLE_NAMES});missing_estimates=0
        for trace in traces[12:]:
            action=trace['neural_action']
            phases[action['phase'] if action is not None else 'hold_missing_estimate']+=1
            visible=np.asarray(trace['adaptation']['role_valid'],dtype=bool)
            if visible.shape!=(3,):raise ValueError('public role visibility shape changed')
            for role,present in zip(ROLE_NAMES,visible):missing[role]+=int(not present)
            missing_estimates+=int(trace['estimate'] is None)
        rows.append({'cell':cell,'trace':binding(directory/'TRACE.npy'),
                     'result':accepted_row['result_binding'],'windows':windows,
                     'phase_occupancy_postprefix':dict(phases),
                     'mask_missing_postprefix':dict(missing),'missing_estimates_postprefix':missing_estimates,
                     'postprefix_steps':228})
    groups=[]
    for split in ('id','geometry'):
        for arm in ('fixed','translation_size5'):
            selected=[r for r in rows if r['cell']['split']==split and r['cell']['arm']==arm]
            if len(selected)!=4:raise ValueError('diagnostic stratum denominator drift')
            windows={}
            for window in WINDOWS:
                values=[r['windows'][window] for r in selected]
                windows[window]={'condition_denominator':4,
                    'conditions_with_estimate':sum(v['valid_estimates']>0 for v in values),
                    'valid_frame_estimates':sum(v['valid_estimates'] for v in values),
                    'expected_frame_estimates':sum(v['expected_estimates'] for v in values),
                    'equal_condition_mean':{key:float(np.mean([v['mean'][key] for v in values if v['mean'][key] is not None]))
                        if any(v['mean'][key] is not None for v in values) else None for key in METRICS}}
            groups.append({'split':split,'arm':arm,'condition_denominator':4,'windows':windows,
                'phase_occupancy_postprefix':dict(sum((Counter(r['phase_occupancy_postprefix']) for r in selected),Counter())),
                'mask_missing_postprefix':{role:sum(r['mask_missing_postprefix'][role] for r in selected) for role in ROLE_NAMES},
                'missing_estimates_postprefix':sum(r['missing_estimates_postprefix'] for r in selected),
                'postprefix_frame_denominator':4*228})
    return {'schema':'prove.p62-postseal-private-geometry-diagnostic.v1',
        'acceptance':binding(acceptance_path),'freeze':binding(freeze_path),'diagnostic_source':binding(__file__),
        'all16_completed_seals_revalidated_before_geometry_truth':True,
        'geometry_truth':'frozen evaluator configuration: center, xyz Euler angles in degrees, half+clearance opening half-size',
        'error_convention':'estimate minus truth; local error in true slot frame; depth signed along true slot +z',
        'windows':{'t12':[12],'t24':[24],'t29':[29],'last10':[230,239],'postprefix':[12,239]},
        'aggregation':'mean within condition, then equal-weight mean across available condition estimates; missing denominators explicit',
        'rows':rows,'groups':groups,'private_postseal_diagnostic_only':True,
        'candidate_updates':0,'selection_changes':0,'new_physics_steps':0,'new_training_steps':0,
        'independent_test_claim':False,'offline_teacher_conditioned_errors_are_not_physical_success':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report=diagnostic(args.acceptance);dump_new(args.output,report)
    print(json.dumps({'rows':len(report['rows']),'groups':report['groups']},indent=2,allow_nan=False))


if __name__=='__main__':main()
