"""Task-bound fixed-z bundles; no cup name/node-count or ROOT_TASK_PLAN dependency."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from .agent import digest
from .io import SafetyError

REPO=Path(__file__).resolve().parents[2]
INVENTORY=REPO/'evaluations/multitask_runtime_v20/TASK_ASSET_INVENTORY.json'


def load_bundle(task,*,inventory=INVENTORY):
    inventory=Path(inventory);catalog=json.loads(inventory.read_text());rows=[r for r in catalog['tasks'] if r['task']==task]
    if len(rows)!=1:raise SafetyError('exact qualified task bundle required: '+task)
    row=rows[0];solution=Path(row['solution']);paths={key:Path(row['bindings'][name]['path']) for key,name in [('graph','graph'),('model','model'),('references','visible_refs'),('manifest','manifest'),('candidate_freeze','candidate_freeze')]}
    for name,binding in row['bindings'].items():
        if digest(binding['path'])!=binding['sha256']:raise SafetyError('task inventory source changed: '+name)
    manifest=json.loads(paths['manifest'].read_text());freeze=json.loads(paths['candidate_freeze'].read_text());graph=json.loads(paths['graph'].read_text());refs=json.loads(paths['references'].read_text())
    if any(v.get('task') not in (task,'twin_'+task) for v in (manifest,freeze,graph,refs)):raise SafetyError('task graph/model/reference identity mismatch')
    for key,fkey in [('graph','graph_sha256'),('model','model_sha256'),('references','references_sha256')]:
        if digest(paths[key])!=freeze[fkey]:raise SafetyError('frozen task asset mismatch: '+key)
    source=Path(row['historical_runner']['path']).parents[2]/'prove/fast_adapter/twin_local_skill.py'
    if digest(source)!=freeze['code']['prove/fast_adapter/twin_local_skill.py']:raise SafetyError('frozen task skill code changed')
    paths['skill_source']=source
    if 'verified_bindings' in manifest:
        motor=next(r for r in manifest['verified_bindings']['sources'] if r['label']=='motor_targets.npz')
    else:
        paths['external_bindings']=solution/'EXTERNAL_BINDINGS.json'
        external=json.loads(paths['external_bindings'].read_text());motor=external['sources']['motor_targets.npz']
    paths['motor_prior']=Path(motor['path'])
    if paths['motor_prior'].resolve()!=Path(row['physics_report']['path']).with_name('motor_targets.npz').resolve():raise SafetyError('motor prior belongs to another physical task')
    if digest(paths['motor_prior'])!=motor['sha256']:raise SafetyError('task motor prior changed')
    training=manifest['model']['training']
    if training['input_dim']!=29 or training['output_dim']!=6:raise SafetyError('this executor requires declared frozen 29D geometry/6D residual contract')
    spec=importlib.util.spec_from_file_location('_multitask_skill_'+task,source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    skill=module.TwinLocalSkill(paths['model'],online_z=False,feedback_contract=graph['local_feedback_contract'])
    if skill.model.input_dim!=training['input_dim']:raise SafetyError('loaded model differs from declared feature contract')
    with np.load(paths['motor_prior'],allow_pickle=False) as data:times=data['timestamps'].copy();q=data['joints'].copy();g=data['gripper'].copy()
    if q.shape!=(len(times),2,6) or g.shape!=(len(times),2) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(g)) or not np.all(np.diff(times)>0) or np.any(g<0) or np.any(g>1):raise SafetyError('task motor prior units/shape/time invalid')
    frames=sorted(refs['frames'].values(),key=lambda f:f['time_sec'])
    if not frames or not graph['stages'] or len({s['stage_id'] for s in graph['stages']})!=len(graph['stages']):raise SafetyError('task stages/reference coverage missing')
    for stage in graph['stages']:
        a=stage['reference']['start_time_sec'];b=stage['reference']['end_time_sec']
        if not times[0]<=a<b<=times[-1]+1e-8:raise SafetyError('stage outside task motor prior')
    return {'task':task,'graph':graph,'refs':refs,'skill':skill,'paths':paths,'bindings':{str(p):digest(p) for p in paths.values()}|{str(inventory.resolve()):digest(inventory)},'times':times,'qcurve':PchipInterpolator(times,q,axis=0),'gripper':g,'initial_q':q[0],'initial_g':g[0],'frames':frames,'frame_times':np.asarray([r['time_sec'] for r in frames]),'dataset_metadata':training['dataset_metadata']}
