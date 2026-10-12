"""Task-specific prepared portable bundle; hardware activation needs strict modern pass."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from prove.xtrainer.agent import digest
from prove.xtrainer.io import SafetyError

SCHEMA='graph-self-adapter.ordinary-real-package.v21'


def check(root,task):
    root=Path(root).resolve();manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('task')!=task or manifest.get('fixed_z') is not True:raise SafetyError('exact task portable manifest required')
    paths=set()
    for row in manifest['files']:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or path in paths or digest(path)!=row['sha256']:raise SafetyError('portable task source/data binding changed')
        paths.add(path)
    runtime=hashlib.sha256(''.join(digest(root/p) for p in manifest['runtime_relative_paths']).encode()).hexdigest()
    if runtime!=manifest['runtime_sha256']:raise SafetyError('portable motion runtime identity mismatch')
    receipt_path=root/'PROMOTION_RECEIPT.json';promoted=False
    if receipt_path.exists():
        expected=(root/'local/ROOT_RECEIPT_SHA256.txt').read_text().strip()
        if digest(receipt_path)!=expected:raise SafetyError('promotion receipt hash mismatch')
        receipt=json.loads(receipt_path.read_text())
        promoted=bool(receipt.get('strict_success') is True and receipt.get('task')==task and receipt.get('runtime_sha256')==runtime and receipt.get('package_manifest_sha256')==digest(root/'PACKAGE_MANIFEST.json'))
        if not promoted:raise SafetyError('same-task modern complete simulation promotion absent')
    return manifest,{'task':task,'runtime_sha256':runtime,'package_verified':True,'promotion_verified':promoted,'devices_opened':0,'hardware_commands':0,'key_read':False,'physical_hardware_qualified':False}


def load_bundle(root,task):
    root=Path(root);manifest,report=check(root,task);paths={key:root/value for key,value in manifest['assets'].items()}
    parent=json.loads(paths['parent_graph'].read_text());route=json.loads(paths['graph'].read_text());freeze=json.loads(paths['candidate_freeze'].read_text());refs=json.loads(paths['references'].read_text())
    from multitask_ordinary_route import ordinary_route
    if route!=ordinary_route(parent) or parent['task'] not in (task,'twin_'+task):raise SafetyError('portable derived task graph mismatch')
    for key,expected in [('parent_graph',freeze['graph_sha256']),('model',freeze['model_sha256']),('references',freeze['references_sha256']),('skill_source',freeze['code']['prove/fast_adapter/twin_local_skill.py'])]:
        if digest(paths[key])!=expected:raise SafetyError('portable frozen task asset differs: '+key)
    spec=importlib.util.spec_from_file_location('_ordinary_portable_skill_'+task,paths['skill_source']);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    skill=module.TwinLocalSkill(paths['model'],online_z=False,feedback_contract=parent['local_feedback_contract'])
    if skill.model.input_dim!=29:raise SafetyError('task frozen 29D feature contract required')
    with np.load(paths['motor_prior'],allow_pickle=False) as data:times=data['timestamps'].copy();q=data['joints'].copy();g=data['gripper'].copy()
    if q.shape!=(len(times),2,6) or g.shape!=(len(times),2) or np.any(np.diff(times)<=0):raise SafetyError('portable motor prior shape/time')
    frames=sorted(refs['frames'].values(),key=lambda f:f['time_sec'])
    return {'task':task,'graph':route,'refs':refs,'skill':skill,'paths':paths,'bindings':{str(root/r['path']):r['sha256'] for r in manifest['files']},'times':times,'qcurve':PchipInterpolator(times,q,axis=0),'gripper':g,'initial_q':q[0],'initial_g':g[0],'frames':frames,'frame_times':np.array([f['time_sec'] for f in frames])},report
