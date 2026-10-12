#!/usr/bin/env python3
"""Isolated ordinary server derivative: fair IO, mapped bundle, optional unused table."""
import argparse,json,os,sys,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (HERE,REPO):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import multitask_ordinary_entry_v21 as base
import multitask_ordinary_runtime as frozen_runtime
from ordinary_server_bundle_v22 import load_bundle
from multitask_ordinary_route import ordinary_route
from prove.xtrainer.io import SafetyError,load_calibration
from prove.xtrainer.agent import digest
TASKS=('collect_apple','collect_cubes','put_knife_board_new','sort_two_spoon')
_installed=False
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def optional_aux_roles(graph,refs):
    from multitask_ordinary_route import role_contract
    result=role_contract(graph,refs);actual={s['object'] for s in graph['stages']}|{s['target'] for s in graph['stages']}
    if graph['task'] not in TASKS:raise SafetyError('ordinary optional-aux audit is task-specific')
    if 'table_patch' not in actual:result['SAM_roles']=[r for r in result['SAM_roles'] if r!='table_patch'];result['optional_SAM_roles']=['table_patch'];result['aux_role_contract']='table patch unused by these four Code/NN/holding/exact-goal routes; no geometry fabricated'
    return result


def install():
    global _installed
    if _installed:return
    _installed=True
    original_load=base.backend.load
    def loader(path,name):
        if Path(path).name=='sim_robot_io.py':
            import fair_sim_robot_io
            return fair_sim_robot_io
        return original_load(path,name)
    base.bindings=bindings;
    base.load_bundle=load_bundle;base.role_contract=optional_aux_roles;frozen_runtime.role_contract=optional_aux_roles;base.backend.load=loader
    base.SOURCES=base.SOURCES+(Path(__file__),HERE/'ordinary_server_bundle_v22.py',HERE/'fair_sim_robot_io.py',REPO/'tests/test_fair_sim_robot_io.py',REPO/'tests/test_ordinary_server_fair_v22.py')


def bindings(config_path,config,mode):
    task=config['task'];ready=base.bundle_check(task)
    b=load_bundle(task);route=Path(config['ordinary_route_graph']);actual=json.loads(route.read_text())
    if actual!=ordinary_route(b['graph']):raise SafetyError('exact ordinary task route differs')
    b['graph']=actual;b['paths']['graph']=route;runtime=copy.deepcopy(config['runtime'])
    if runtime.get('hardware_reference_speed_scale')!=1. or len(runtime.get('vision',{}).get('role_boxes',{}))!=0:raise SafetyError('fresh initial grounding and reference clock1.0 required')
    from multitask_semantic_judge import validate_config
    validate_config(runtime['stage_judge'])
    if runtime['stage_judge']['maximum_calls']!=ready['required_supported_judgements']:raise SafetyError('exact declared task semantic budget required')
    camera,right=load_calibration(runtime['left_calibration'],runtime['right_calibration']);runtime.update(T_left_from_camera=camera.tolist(),T_left_from_right=right.tolist())
    paths=list(base.SOURCES)+[config_path,route,Path(config['formal_run_reservation']),Path(config['sim']['physics_source']),Path(runtime['sdk_file']),Path(runtime['left_calibration']),Path(runtime['right_calibration']),Path(runtime['vision']['sam_utility']),Path(runtime['vision']['sam_checkpoint'])]
    scene=Path(config['sim']['scene_run'])
    for name in ('scene.xml','reconstruction.json','motor_targets.npz','reference.npz','dynamics.npz','physics_validation.json'):paths.append(scene/name)
    # Bind the historical task's full external-asset catalog, without using evaluator truth at runtime.
    catalog=REPO/'evaluations/multitask_parallel_v22/server_shared/EXTERNAL_ASSET_BINDINGS_V01_SERVER.json';paths.append(catalog)
    asset_catalog=json.loads(catalog.read_text());rows=asset_catalog['task_assets'][task]['external_asset_hashes']
    for evaluator in asset_catalog.get('external_evaluator_sources',[]):
        if digest(evaluator['path'])!=evaluator['sha256']:raise SafetyError('postseal evaluator source changed')
        paths.append(Path(evaluator['path']))
    for path,sha in rows.items():
        if digest(path)!=sha:raise SafetyError('task scene asset changed')
        paths.append(Path(path))
    paths.append(REPO/'evaluations/multitask_parallel_v22/server_shared/REMOTE_DERIVATIVE_MANIFEST.json')
    bound=dict(b['bindings']);bound.update({str(Path(path).resolve()):digest(path) for path in paths})
    return {'dependencies':{}},runtime,b,bound

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv);p=argparse.ArgumentParser(add_help=False);p.add_argument('--task',required=True);a,_=p.parse_known_args(args)
    if a.task not in TASKS:raise SafetyError('ordinary server task not owned')
    check_only='check' in args or not any(mode in args for mode in ('sim','shadow','real'))
    if not check_only and (not os.environ.get('CUDA_VISIBLE_DEVICES') or any(os.environ.get(key)!='1' for key in THREAD_ENV)):raise SafetyError('server requires explicit CUDA and four BLAS/Torch thread env limits1')
    install();return base.main(args)

if __name__=='__main__':main()
