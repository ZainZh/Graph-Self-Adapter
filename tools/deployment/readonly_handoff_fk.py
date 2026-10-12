#!/usr/bin/env python3
"""Three FK-only queries for historical neural handoff geometry; no commands."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation


def finite(value,shape,label):
    array=np.asarray(value,dtype=float)
    if array.shape!=shape or not np.isfinite(array).all():raise ValueError('invalid '+label)
    return array


def inputs(diagnostic,entry,bundle):
    diag=json.loads(Path(diagnostic).read_text());actual=json.loads(Path(entry).read_text())
    if diag.get('offline_only') is not True or diag.get('NoMotion') is not True or actual.get('NoMotion') is not True:
        raise ValueError('only declared offline/no-motion diagnostic inputs accepted')
    if not diag.get('stage_id','').endswith('.grasp'):raise ValueError('expected first grasp neural diagnostic')
    plan=actual['candidate_plan']
    endpoint=finite(plan['q_path'][-1],(2,6),'historical entry endpoint')
    neural=finite(diag['neural_target_q_ref_plus_residual_rad'],(2,6),'first neural absolute target')
    reference=finite(bundle['qcurve'](float(diag['reference_query_time_sec'])),(2,6),'query-time reference')
    if not np.allclose(endpoint,finite(diag['actual_entry_q_rad'],(2,6),'diagnostic entry'),atol=1e-10):
        raise ValueError('entry endpoint differs from neural diagnostic')
    # Reconstruct the exact SHADOW point used in the diagnostic from its recorded
    # first-Approach source delta and immutable corresponding visible reference.
    start=bundle['graph']['stages'][0]['reference']['start_time_sec']
    frame=bundle['frames'][int(np.argmin(abs(bundle['frame_times']-start)))]
    red=finite(frame['surface_centroids_world_m']['red_mug'],(3,),'Approach reference red point')+finite(diag['source_delta_m'],(3,),'SHADOW source delta')
    offset=finite(plan['declared_reference_object_TCP_offset_m'],(3,),'declared entry offset')
    entry_rotation=finite(plan['goal_rotation_shared'],(3,3),'declared entry orientation')
    return diag,actual,[endpoint,neural,reference],red,offset,entry_rotation


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--entry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('fresh handoff FK output required')
    package=a.package.resolve();sys.path.insert(0,str(package))
    spec=importlib.util.spec_from_file_location('_readonly_handoff_cli',package/'tools/task_graph/xtrainer_portable.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    cfg_path=a.config or package/'local/CONFIG.json';cfg=json.loads(cfg_path.read_text())
    check,cfg,bundle=cli.portable_check(package,cfg)
    if check['missing_external_files'] or any(not check['dependencies'].get(name) for name in ('numpy','scipy','PIL','yaml','zmq')):
        raise ValueError('required FK dependencies missing before connection')
    diag,entry,targets,red,offset,goal_rotation=inputs(a.diagnostic,a.entry,bundle)
    T=finite(cfg['T_left_from_right'],(4,4),'left-from-right transform')
    g=finite(cfg['known_initial_gripper_commands'],(2,),'expected command for FK packing')
    report={'schema':'graph-self-adapter.readonly-neural-handoff-FK.v1','NoMotion':True,'camera_opened':False,
        'hardware_motion_commands':0,'observation_cached':True,'live_admission':False,
        'red_object_shared_m':red.tolist(),'red_source':'exact historical SHADOW point reconstructed from bound first-Approach delta',
        'entry_declared_offset_m':offset.tolist(),'entry_declared_orientation_shared':goal_rotation.tolist(),
        'FK_pack_gripper_command_not_measurement':g.tolist(),'RPC_timeout_ms':1000,'maximum_FK_queries':3,
        'queries':[],'bindings':{str(path.resolve()):cli.digest(path) for path in (a.diagnostic,a.entry,cfg_path,Path(__file__))},
        'limitations':['historical object point; no live motion admission','no collision check or smoothing execution','frozen delta-q-reference semantics unchanged']}
    a.output.mkdir(parents=True);save=lambda:cli.write(a.output/'RESULT.json',report);save()
    class FKOnly(cli.NativeRPC):
        def __init__(self):super().__init__(timeout_ms=1000);self.query_count=0
        def call(self,method,**kwargs):
            if method!='get_fk':raise cli.SafetyError('only get_fk permitted')
            if self.query_count>=3:raise cli.SafetyError('three-FK query budget exhausted')
            self.query_count+=1;return super().call(method,**kwargs)
        def command(self,*args,**kwargs):raise cli.SafetyError('command unavailable in read-only FK diagnosis')
    rpc=FKOnly()
    try:
        rpc.connect()
        for name,q in zip(('entry_endpoint','first_neural_absolute_target','reference_at_query_time'),targets):
            row={'name':name,'q_rad':q.tolist()};report['queries'].append(row);save()
            values=finite(rpc.call('get_fk',joint_state=rpc.pack(q,g)),(12,),'native FK response')
            native_left=values[:3];left_rotation=Rotation.from_rotvec(values[3:6]).as_matrix()
            native_right=values[6:9];right_rotation_native=Rotation.from_rotvec(values[9:12]).as_matrix()
            right_shared=(T@np.r_[native_right,1])[:3];right_rotation_shared=T[:3,:3]@right_rotation_native
            current_offset=native_left-red;xyz_error=current_offset-offset
            rot_error=Rotation.from_matrix(left_rotation@goal_rotation.T).as_rotvec()
            row.update(left_TCP_shared_m=native_left.tolist(),left_rotation_shared=left_rotation.tolist(),
                left_object_TCP_offset_m=current_offset.tolist(),offset_error_xyz_m=xyz_error.tolist(),
                offset_error_norm_m=float(np.linalg.norm(xyz_error)),orientation_error_rotvec_rad=rot_error.tolist(),
                orientation_error_rad=float(np.linalg.norm(rot_error)),right_TCP_native_m=native_right.tolist(),
                right_TCP_shared_m=right_shared.tolist(),right_rotation_shared=right_rotation_shared.tolist());save()
        first=report['queries'][0]
        for row in report['queries'][1:]:
            row['right_native_TCP_delta_from_entry_m']=(np.asarray(row['right_TCP_native_m'])-first['right_TCP_native_m']).tolist()
            row['right_native_TCP_distance_from_entry_m']=float(np.linalg.norm(row['right_native_TCP_delta_from_entry_m']))
            row['left_TCP_delta_from_entry_m']=(np.asarray(row['left_TCP_shared_m'])-first['left_TCP_shared_m']).tolist()
        report['completed_queries']=rpc.query_count;report['geometric_entry_qualified']=None
        report['qualification_note']='Root must assess visible relation and the recorded pose/offset errors; no automatic motion acceptance.'
        save();return report
    except Exception as exc:report.update(error=str(exc),partial_log_preserved=True);save();return report
    finally:rpc.close()

if __name__=='__main__':main()
