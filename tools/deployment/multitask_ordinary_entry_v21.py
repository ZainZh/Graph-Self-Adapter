#!/usr/bin/env python3
"""Thin task-bound ordinary sim/real entry; default check never opens I/O."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for p in (HERE,REPO):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from prove.xtrainer.agent import digest
from prove.xtrainer.io import SafetyError,load_calibration
from prove.xtrainer.multitask_bundle import load_bundle,INVENTORY
from multitask_ordinary_route import ordinary_route,role_contract
import direct_white_entry as backend

SOURCES=(Path(__file__),HERE/'multitask_ordinary_runtime_v21.py',HERE/'multitask_ordinary_runtime.py',HERE/'multitask_ordinary_route.py',HERE/'multitask_semantic_judge.py',HERE/'compact_multitask_judge.py',HERE/'object_centered_runtime.py',HERE/'rgbd_capture_retry.py',HERE/'shared_cup_vision.py',HERE/'sim_robot_io.py',HERE/'operator_launcher.py',HERE/'real_feedback_stream.py',HERE/'compact_stage_judge.py',HERE/'direct_white_entry.py',REPO/'prove/xtrainer/multitask_bundle.py',REPO/'prove/xtrainer/runtime.py',REPO/'prove/xtrainer/io.py',REPO/'prove/xtrainer/agent.py',REPO/'prove/xtrainer/perception.py',REPO/'prove/task_graph/twin_sensor_adapter.py',REPO/'tests/test_multitask_ordinary.py',REPO/'tests/test_multitask_relative_lift_v21.py',REPO/'evaluations/multitask_runtime_v20/ROOT_TASK_ROUTES.json')


def runtime_hash():return hashlib.sha256(''.join(digest(path) for path in SOURCES).encode()).hexdigest()


def bundle_check(task):
    b=load_bundle(task);graph=ordinary_route(b['graph']);roles=role_contract(graph,b['refs'])
    return {'task':task,'parent_graph_stages':len(b['graph']['stages']),'routed_stages':len(graph['stages']),'declared_model_dim':b['skill'].model.input_dim,'fixed_z':True,'roles':roles,'required_supported_judgements':2*sum(s['phase']=='Grasp' for s in graph['stages']),'runtime_sha256':runtime_hash(),'device_connections':0,'api_calls':0,'episode_started':False,'hardware_qualified':False}


def bindings(config_path,config,mode):
    task=config['task'];ready=bundle_check(task)
    b=load_bundle(task);route=Path(config['ordinary_route_graph']);actual=json.loads(route.read_text())
    if actual!=ordinary_route(b['graph']):raise SafetyError('exact ordinary task route differs')
    b['graph']=actual;b['paths']['graph']=route;runtime=copy.deepcopy(config['runtime'])
    if runtime.get('hardware_reference_speed_scale')!=1. or len(runtime.get('vision',{}).get('role_boxes',{}))!=0:raise SafetyError('fresh initial grounding and reference clock1.0 required')
    from multitask_semantic_judge import validate_config
    validate_config(runtime['stage_judge'])
    if runtime['stage_judge']['maximum_calls']!=ready['required_supported_judgements']:raise SafetyError('exact declared task semantic budget required')
    camera,right=load_calibration(runtime['left_calibration'],runtime['right_calibration']);runtime.update(T_left_from_camera=camera.tolist(),T_left_from_right=right.tolist())
    paths=list(SOURCES)+[config_path,route,Path(config['formal_run_reservation']),Path(config['sim']['physics_source']),Path(runtime['sdk_file']),Path(runtime['left_calibration']),Path(runtime['right_calibration']),Path(runtime['vision']['sam_utility']),Path(runtime['vision']['sam_checkpoint'])]
    scene=Path(config['sim']['scene_run'])
    for name in ('scene.xml','reconstruction.json','motor_targets.npz','reference.npz','dynamics.npz','physics_validation.json'):paths.append(scene/name)
    # Bind the historical task's full external-asset catalog, without using evaluator truth at runtime.
    catalog=REPO/'evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json';paths.append(catalog)
    asset_catalog=json.loads(catalog.read_text());rows=asset_catalog['task_assets'][task]['external_asset_hashes']
    for evaluator in asset_catalog.get('external_evaluator_sources',[]):
        if digest(evaluator['path'])!=evaluator['sha256']:raise SafetyError('postseal evaluator source changed')
        paths.append(Path(evaluator['path']))
    for path,sha in rows.items():
        if digest(path)!=sha:raise SafetyError('task scene asset changed')
        paths.append(Path(path))
    bound=dict(b['bindings']);bound.update({str(Path(path).resolve()):digest(path) for path in paths})
    return {'dependencies':{}},runtime,b,bound

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('mode',choices=('check','sim','shadow','real'),nargs='?',default='check')
    for name in ('config','scope','execution-freeze','output'):p.add_argument('--'+name,type=Path)
    a=p.parse_args(argv);ready=bundle_check(a.task)
    if a.mode=='check':print(json.dumps(ready,indent=2));return ready
    if not all((a.config,a.scope,a.execution_freeze,a.output)):raise SafetyError('explicit new task config/scope/freeze/output required')
    if a.mode in ('shadow','real'):
        # This generic candidate has not yet earned a portable physical activation receipt.
        scope=json.loads(a.scope.read_text());gate=scope.get('passed_simulation_report')
        if not gate or digest(gate['path'])!=gate['sha256']:raise SafetyError('same-runtime whole-task simulation receipt required before real I/O')
        evidence=json.loads(Path(gate['path']).read_text())
        if evidence.get('strict_success') is not True or evidence.get('shared_runtime_sha256')!=runtime_hash() or evidence.get('task')!=a.task:raise SafetyError('modern task promotion absent')
        raise SafetyError('portable real activation adapter must be frozen after this task simulation passes; no cup fallback')
    cfg=json.loads(a.config.read_text())
    if cfg.get('task')!=a.task:raise SafetyError('configuration belongs to another task')
    original_load=backend.load
    def load(path,name):
        if Path(path).name=='direct_white_runtime.py':
            from multitask_ordinary_runtime_v21 import RelativeLiftExecutor,SharedSAMVision
            return SimpleNamespace(SharedCupExecutor=RelativeLiftExecutor,SharedSAMVision=SharedSAMVision)
        return original_load(path,name)
    from multitask_semantic_judge import MultitaskSemanticJudge
    def judge(*args,**kwargs):return MultitaskSemanticJudge(*args,**kwargs,key_file=cfg['runtime']['stage_judge']['key_file'],maximum_calls=cfg['runtime']['stage_judge']['maximum_calls'])
    backend.TASK=a.task;backend.VARIANT='multitask_ordinary_object_centered_v20';backend.bindings=bindings;backend.runtime_hash=runtime_hash;backend.load=load;backend.RealRootJudge=judge
    return backend.main(['sim','--config',str(a.config),'--scope',str(a.scope),'--execution-freeze',str(a.execution_freeze),'--output',str(a.output)])

if __name__=='__main__':main()
