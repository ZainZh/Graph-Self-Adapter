"""Relocatable, hash-bound cup package loading; no simulator/project-plan dependency."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from .agent import digest
from .io import SafetyError,load_calibration

SCHEMA='graph-self-adapter.xtrainer-portable-package.v1'
TASK='put_mug_on_box'


def verify_package(root):
    root=Path(root).resolve();manifest_path=root/'PACKAGE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('version')!='xtrainer-laptop-v01' or manifest.get('task')!=TASK or manifest.get('fixed_z') is not True:
        raise SafetyError('unsupported portable cup package')
    bound={}
    for row in manifest['files']:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or path in bound or not path.is_file() or digest(path)!=row['sha256']:
            raise SafetyError('portable path/hash mismatch: '+row['path'])
        bound[path]=row['sha256']
    required=('graph/task_graph.json','models/local_skill.npz','refs/visible_centroids.json',
              'source/frozen_twin_local_skill.py','motor_targets.npz','CANDIDATE_FREEZE.json',
              'prove/xtrainer/io.py','prove/xtrainer/agent.py','prove/xtrainer/runtime.py',
              'prove/xtrainer/perception.py','prove/xtrainer/mock.py','prove/xtrainer/portable.py',
              'tools/task_graph/xtrainer_portable.py','CONFIG_PROPOSED.json','run.sh','README.md')
    if not all(root/path in bound for path in required):raise SafetyError('portable package missing required implementation/assets')
    frozen=json.loads((root/'CANDIDATE_FREEZE.json').read_text())
    expected={'graph/task_graph.json':frozen['graph_sha256'],'models/local_skill.npz':frozen['model_sha256'],
              'refs/visible_centroids.json':frozen['references_sha256'],
              'source/frozen_twin_local_skill.py':frozen['code']['prove/fast_adapter/twin_local_skill.py'],
              'motor_targets.npz':manifest['original_motor_sha256']}
    for relative,sha in expected.items():
        if bound.get(root/relative)!=sha:raise SafetyError('original frozen portable asset changed: '+relative)
    bound[manifest_path]=digest(manifest_path)
    return manifest,bound


def load_portable_bundle(root):
    root=Path(root).resolve();manifest,bound=verify_package(root)
    source=root/'source/frozen_twin_local_skill.py'
    spec=importlib.util.spec_from_file_location('_portable_A2_frozen_skill',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    paths={'graph':root/'graph/task_graph.json','model':root/'models/local_skill.npz',
           'references':root/'refs/visible_centroids.json','skill_source':source,
           'motor_prior':root/'motor_targets.npz','candidate_freeze':root/'CANDIDATE_FREEZE.json'}
    graph=json.loads(paths['graph'].read_text());refs=json.loads(paths['references'].read_text())
    with np.load(paths['motor_prior'],allow_pickle=False) as data:
        times=data['timestamps'].copy();q=data['joints'].copy();g=data['gripper'].copy()
    if q.shape!=(len(times),2,6) or g.shape!=(len(times),2) or not np.all(np.diff(times)>0):raise SafetyError('portable motor prior shape/time mismatch')
    skill=module.TwinLocalSkill(paths['model'],online_z=False,feedback_contract=graph['local_feedback_contract'])
    if skill.model.input_dim!=29 or len(graph['stages'])!=28:raise SafetyError('unexpected portable cup graph/model')
    frames=sorted(refs['frames'].values(),key=lambda f:f['time_sec'])
    return {'graph':graph,'refs':refs,'skill':skill,'paths':paths,'bindings':{str(p):sha for p,sha in bound.items()},
            'times':times,'qcurve':PchipInterpolator(times,q,axis=0),'gripper':g,'initial_q':q[0],'initial_g':g[0],
            'frame_times':np.array([x['time_sec'] for x in frames]),'frames':frames}


def rebind_config(config,*,robot_project=None,sam_root=None,sam_checkpoint=None):
    cfg=copy.deepcopy(config)
    if robot_project:
        project=Path(robot_project).expanduser().resolve()
        cfg['sdk_file']=str(project/'include/xtrainer_clover/dobot_control/robots/dobot_api.py')
        for arm in ('left','right'):
            cfg[arm+'_calibration']=str(project/f'script/real_zed_collection/calibration/robot_camera_apriltag_{arm}_global.yaml')
        cfg['vision']['sam_utility']=str(project/'script/real_zed_collection/sam2_tracking_utils.py')
    for key,value in (('sam_root',sam_root),('sam_checkpoint',sam_checkpoint)):
        if value:cfg['vision'][key]=str(Path(value).expanduser().resolve())
    return cfg


def nominal_envelopes(bundle,cfg):
    dt=cfg['command_dt_sec'];scale=cfg['hardware_reference_speed_scale']
    if not 0<scale<=1:raise SafetyError('invalid portable reference speed scale')
    rows=[]
    for stage in bundle['graph']['stages']:
        start=stage['reference']['start_time_sec'];end=stage['reference']['end_time_sec']
        step=dt*stage.get('reference_speed_scale',1.)*scale
        times=np.r_[np.arange(start,end,step),end];q=bundle['qcurve'](times)
        peak=float(np.max(abs(np.diff(q,axis=0))))
        rows.append({'stage_id':stage['stage_id'],'max_step_rad':peak,'max_velocity_rad_sec':peak/dt,
            'passed':peak<=cfg['safety']['max_step_rad']+1e-9 and peak/dt<=cfg['safety']['max_velocity_rad_sec']+1e-9})
    return rows


def portable_check(root,config):
    bundle=load_portable_bundle(root);cfg=copy.deepcopy(config);gaps=[];external={}
    paths=[cfg['sdk_file'],cfg['left_calibration'],cfg['right_calibration'],cfg['vision']['sam_utility'],cfg['vision']['sam_checkpoint']]
    sam=Path(cfg['vision']['sam_root'])
    paths += [str(sam/'sam2/build_sam.py'),str(sam/'sam2/sam2_camera_predictor.py'),str(sam/'configs'/cfg['vision']['sam_config'])]
    for value in paths:
        path=Path(value).expanduser().resolve()
        if path.is_file():external[str(path)]=digest(path)
        else:gaps.append(str(path))
    if not any(str(Path(cfg[key]).expanduser().resolve()) in gaps for key in ('left_calibration','right_calibration')):
        camera,right=load_calibration(cfg['left_calibration'],cfg['right_calibration'])
        cfg['T_left_from_camera']=camera.tolist();cfg['T_left_from_right']=right.tolist()
    report={'schema':'graph-self-adapter.xtrainer-portable-check.v1','task':TASK,'fixed_z':True,'graph_stages':28,
            'model_input_dim':29,'hardware_connections':0,'hardware_motion_commands':0,'hardware_ready':False,
            'missing_external_files':gaps,'package_bindings':bundle['bindings'],'external_bindings':external,
            'dependencies':{name:importlib.util.find_spec(name) is not None for name in ('numpy','scipy','PIL','yaml','zmq','torch','pyzed')},
            'nominal_envelopes':nominal_envelopes(bundle,cfg),
            'initial_gripper_state_confirmed':cfg['initial_gripper_state_confirmed'],
            'hardware_safety_reviewed':cfg['hardware_safety_reviewed'],
            'limits':'no measured aperture/collision or real task qualification; original Executor safety preserved'}
    report['nominal_plan_within_limits']=all(row['passed'] for row in report['nominal_envelopes'])
    return report,cfg,bundle
