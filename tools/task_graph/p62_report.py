#!/usr/bin/env python3
"""Read-only acceptance audit for the fixed sixteen-cell P62 physical scope.

Call only after every prespecified attempt has ended. This tool does not import
or construct a simulator and never retries cells. Physical success is separate
from the previously opened offline teacher-conditioned action diagnostic.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prove.task_graph.precision_action_policy import load_frozen_head, encode_state, decode_action
from tools.task_graph.p62_visual_validation import summarize, cell_name, binding
from tools.task_graph.train_precision_action_policy import dump_new

ARMS=('fixed','translation_size5')


def _array_error(actual,expected):
    actual,expected=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite or mismatched command/latent array')
    return float(np.max(np.abs(actual-expected),initial=0.))


def trace_audit(directory,result,head):
    traces=np.load(directory/'TRACE.npy',allow_pickle=True).tolist()
    predictions=np.load(directory/'PREDICTIONS.npy',allow_pickle=True).tolist()
    if len(traces)!=240 or len(predictions)!=240 or result['steps']!=240:
        raise ValueError('completed cell lacks all 240 steps')
    if [int(r['t']) for r in traces]!=list(range(240)) or [int(r['t']) for r in predictions]!=list(range(240)):
        raise ValueError('nonconsecutive physical trace')
    length=float(result['tool_calibration']['length'])
    if result['tool_calibration']['source']!='RGB visible-mask bbox plus known own-tool pose':
        raise ValueError('physical held length is not public calibrated')
    rotation_max=latent_max=0.;commands=holds=0
    command_error=action_error=feature_error=decode_error=0.
    before_losses=[];after_losses=[];gradient_steps=0
    for trace,prediction in zip(traces,predictions):
        for key in ('z_before','z_after'):
            z=np.asarray(prediction[key],dtype=float)
            if z.shape!=(8,) or not np.isfinite(z).all():raise ValueError('invalid persistent latent')
            rotation_max=max(rotation_max,float(np.max(np.abs(z[3:6]))))
            latent_max=max(latent_max,float(np.max(np.abs(z))))
            if _array_error(trace['adaptation'][key],z)!=0:raise ValueError('trace/prediction latent disagree')
        gradient_steps+=int(prediction['U'])
        for key,target in (('pre_feedback_loss',before_losses),('post_feedback_loss',after_losses)):
            value=prediction[key]
            if value is not None:
                if not np.isfinite(value):raise ValueError('nonfinite public feedback loss')
                target.append(float(value))
        t=int(trace['t']);action=trace['neural_action'];estimate=trace['estimate']
        if t<12:
            if action is not None:raise ValueError('neural action during common observation prefix')
            continue
        if estimate is None:
            if action is not None:raise ValueError('neural command despite unavailable estimate')
            command_error=max(command_error,_array_error(trace['command_p'],trace['proprio'][0]),
                              _array_error(trace['command_R'],trace['proprio'][1]))
            holds+=1;continue
        if action is None or action['t']!=t or action['command_source']!='frozen_precision_action_mlp':
            raise ValueError('post-prefix command lacks its actual frozen P50 action record')
        features=encode_state(estimate,trace['proprio'],length,action['phase'])
        feature_error=max(feature_error,_array_error(action['features'],features))
        with torch.no_grad():expected=head(torch.as_tensor(features)[None])[0].numpy()
        action_error=max(action_error,_array_error(action['normalized_action'],expected))
        decoded=decode_action(expected,estimate[1],trace['proprio'])
        decode_error=max(decode_error,_array_error(action['position'],decoded[0]),
                         _array_error(action['rotation'],decoded[1]))
        command_error=max(command_error,_array_error(trace['command_p'],action['position']),
                          _array_error(trace['command_R'],action['rotation']))
        commands+=1
    if rotation_max!=0.:raise ValueError('rotation latent was modified')
    if result['cell']['arm']=='fixed' and (latent_max!=0. or gradient_steps!=0):
        raise ValueError('fixed arm updated latent')
    if gradient_steps!=result['gradient_steps']:raise ValueError('gradient-step summary differs from trace')
    if commands==0:raise ValueError('no frozen P50 commands were issued')
    if max(command_error,decode_error,action_error,feature_error)>1e-6:
        raise ValueError('actual command does not reproduce frozen P50 path within 1e-6')
    return {'passed':True,'steps':len(traces),'neural_command_steps':commands,'missing_estimate_hold_steps':holds,
            'gradient_steps':gradient_steps,'max_abs_z_rotation':rotation_max,'max_abs_z':latent_max,
            'P50_feature_max_abs_error':feature_error,'P50_action_max_abs_error':action_error,
            'P50_decoded_pose_max_abs_error':decode_error,'issued_command_max_abs_error':command_error,
            'command_check':'recompute features from estimated geometry and actual proprio, frozen P50 forward, decode, then compare issued pose; no action Oracle',
            'public_feedback_loss':{'count_before':len(before_losses),'count_after':len(after_losses),
                'mean_before':float(np.mean(before_losses)) if before_losses else None,
                'mean_after':float(np.mean(after_losses)) if after_losses else None}}


def build_report(freeze_path):
    freeze_path=Path(freeze_path)
    frozen=json.loads(freeze_path.read_text());summary=summarize(freeze_path)
    if summary['expected']!=16 or summary['completed']+summary['aborted']!=16:
        raise ValueError('all sixteen prespecified attempts must have ended before acceptance')
    expected={(split,seed,arm) for split,base in (('id',611100),('geometry',611200))
              for seed in range(base,base+4) for arm in ARMS}
    actual={(r['cell']['split'],r['cell']['condition_seed'],r['cell']['arm']) for r in summary['cells']}
    if len(summary['cells'])!=16 or actual!=expected:
        raise ValueError('P62 physical denominator differs from frozen eight paired conditions')
    if any(r['cell']['encoder']!='task_trained_lila' for r in summary['cells']):
        raise ValueError('unexpected model arm')
    head_binding=frozen['files']['action_head']
    if binding(head_binding['path'])['sha256']!=head_binding['sha256']:
        raise ValueError('frozen P50 checkpoint changed before command audit')
    head=load_frozen_head(head_binding['path'])
    rows=[]
    for index,item in enumerate(summary['cells']):
        directory=freeze_path.parent/'cells'/cell_name(index,item['cell'])
        if (directory/'RESULT.json').exists() and (directory/'ABORT.json').exists():
            raise ValueError('ambiguous completed and aborted attempt')
        pre=json.loads((directory/'PRE_RUN.json').read_text())
        if pre['cell']!=item['cell'] or pre['freeze']['sha256']!=binding(freeze_path)['sha256']:
            raise ValueError('pre-run condition/freeze binding mismatch')
        row={'cell':item['cell'],'status':item['status'],'directory':str(directory.resolve()),
             'pre_run':binding(directory/'PRE_RUN.json')}
        if item['status']=='complete':
            result=item['result'];row.update(result=result,result_binding=binding(directory/'RESULT.json'),
                seal=binding(directory/'SEAL.json'),native_sustained=bool(result['sustained_native_success']),
                strict_success=bool(result['success']),validated_success=bool(result['validated_success']),
                mechanical_anomaly=not bool(result['mechanical_valid']))
            try:row['trace_audit']=trace_audit(directory,result,head)
            except Exception as exc:row['trace_audit']={'passed':False,'error_type':type(exc).__name__,'reason':str(exc)}
        else:
            if item['abort']['cell']!=item['cell']:raise ValueError('abort condition identity mismatch')
            row.update(abort=item['abort'],abort_binding=binding(directory/'ABORT.json'),
                       native_sustained=False,strict_success=False,validated_success=False,
                       mechanical_anomaly=None,trace_audit={'passed':False,'reason':'aborted; complete trace unavailable'})
        rows.append(row)
    groups=[];pairs=[]
    for split in ('id','geometry'):
        for arm in ARMS:
            selected=[r for r in rows if r['cell']['split']==split and r['cell']['arm']==arm]
            groups.append({'split':split,'arm':arm,'denominator':4,
                'completed':sum(r['status']=='complete' for r in selected),'aborted':sum(r['status']=='abort' for r in selected),
                'native_sustained':sum(r['native_sustained'] for r in selected),
                'strict_success':sum(r['strict_success'] for r in selected),
                'validated_success':sum(r['validated_success'] for r in selected),
                'mechanical_anomalies_completed':sum(r['mechanical_anomaly'] is True for r in selected),
                'mechanical_unknown_aborted':sum(r['mechanical_anomaly'] is None for r in selected)})
        seeds=sorted({r['cell']['condition_seed'] for r in rows if r['cell']['split']==split})
        for seed in seeds:
            pair={r['cell']['arm']:r for r in rows if r['cell']['condition_seed']==seed}
            fixed,online=[pair[a]['validated_success'] for a in ARMS]
            outcome='rescue' if online and not fixed else 'harm' if fixed and not online else 'tie_success' if online else 'tie_failure'
            pairs.append({'split':split,'condition_seed':seed,'outcome':outcome,
                'fixed_validated':fixed,'online_validated':online,'has_abort':any(r['status']=='abort' for r in pair.values()),
                'steps':{a:pair[a].get('result',{}).get('steps',pair[a].get('abort',{}).get('completed_steps')) for a in ARMS}})
    return {'schema':'prove.p62-physical-acceptance-report.v1','freeze':binding(freeze_path),
        'frozen_p50':head_binding,'reporter':binding(__file__),'expected_cells':16,'completed':summary['completed'],'aborted':summary['aborted'],
        'all_attempts_ended':True,'all_complete_trace_audits_pass':all(r['trace_audit']['passed'] for r in rows if r['status']=='complete'),
        'all_cells_engineering_valid':all(r['trace_audit']['passed'] for r in rows),
        'groups':groups,'pairs':pairs,'paired_counts':{k:sum(p['outcome']==k for p in pairs) for k in ('rescue','harm','tie_success','tie_failure')},
        'paired_counts_complete_attempts_only':{k:sum(p['outcome']==k and not p['has_abort'] for p in pairs)
                                                for k in ('rescue','harm','tie_success','tie_failure')},
        'rows':rows,'offline_results_are_not_physical_success':True,'fresh_test_claim':False,
        'scope':'eight bounded physical development conditions; one frozen task-trained encoder; all attempts retained',
        'abort_handling':'aborts count in denominators, success false; mechanical status unknown, never silently dropped'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report=build_report(args.freeze);dump_new(args.output,report)
    print(json.dumps({k:report[k] for k in ('expected_cells','completed','aborted','groups','paired_counts','all_cells_engineering_valid')},indent=2))


if __name__=='__main__':main()
