#!/usr/bin/env python3
"""One object-centered graph/Code/NN/SAM/Root pipeline, thin sim/real I/O only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from prove.xtrainer.portable import portable_check
from prove.xtrainer.agent import RealRootJudge,digest,write
from prove.xtrainer.io import SafetyError
if str(Path(__file__).resolve().parent) not in sys.path:sys.path.insert(0,str(Path(__file__).resolve().parent))

HERE=Path(__file__).resolve().parent
VARIANT='handle_macro_continuous_transfer_v1'
TASK='put_mug_on_box'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def bindings(config_path,cfg,mode):
    from handle_macro_route import macro_route_graph
    if mode!='sim':raise SafetyError('unpromoted macro engineering scope is simulator-only')
    root=Path(cfg['portable_package']).resolve();selected_runtime=cfg['runtime'] if mode=='sim' else json.loads(Path(cfg['real_operator_config']).read_text())
    if selected_runtime.get('hardware_reference_speed_scale')!=1.0:raise SafetyError('shared sim/real reference speed must be 1.0')
    report,effective,bundle=portable_check(root,selected_runtime)
    paths={HERE/'shared_cup_entry.py',HERE/'shared_cup_runtime.py',HERE/'object_centered_runtime.py',HERE/'real_feedback_stream.py',HERE/'operator_launcher.py',
           HERE/'sim_robot_io.py',HERE/'shared_cup_vision.py',config_path.resolve(),REPO/'prove/xtrainer/perception.py',REPO/'prove/xtrainer/agent.py',
           REPO/'prove/xtrainer/runtime.py',REPO/'prove/xtrainer/io.py',REPO/'prove/task_graph/twin_sensor_adapter.py'}
    bound={**bundle['bindings'],**report['external_bindings']}
    if selected_runtime.get('static_reference_points_file') is not None:
        prior=Path(selected_runtime['static_reference_points_file'])
        expected=selected_runtime.get('static_reference_points_sha256')
        if expected is not None and digest(prior)!=expected:raise SafetyError('paired static reference binding mismatch')
        paths.add(prior)
        for row in json.loads(prior.read_text()).get('evidence_bindings',[]):
            if digest(row['path'])!=row['sha256']:raise SafetyError('paired reference evidence changed')
            paths.add(Path(row['path']))
    if mode=='real':paths.add(Path(cfg['real_operator_config']))
    if mode=='sim':
        physics=Path(cfg['sim']['physics_source']);scene=Path(cfg['sim']['scene_run'])
        paths.add(physics)
        for name in ('scene.xml','reconstruction.json','reference.npz','motor_targets.npz','physics_validation.json','dynamics.npz'):paths.add(scene/name)
        catalog=REPO/'evaluations/twin_rsi_v02/EXTERNAL_ASSET_BINDINGS.json';paths.add(catalog)
        catalog_value=json.loads(catalog.read_text())
        assets=catalog_value['task_assets'][TASK]['external_asset_hashes']
        for evaluator in catalog_value.get('external_evaluator_sources',[]):
            if digest(evaluator['path'])!=evaluator['sha256']:raise SafetyError('post-seal evaluator source changed')
            paths.add(Path(evaluator['path']))
        for path,sha in assets.items():
            if digest(path)!=sha:raise SafetyError('original scene asset changed: '+path)
            paths.add(Path(path))
    route=Path(cfg['macro_route_graph']);routed=json.loads(route.read_text())
    if routed!=macro_route_graph(bundle['graph']):raise SafetyError('exact declared white Code route differs')
    bundle['graph']=routed;bundle['paths']['graph']=str(route.resolve())
    paths.add(Path(cfg['formal_run_reservation']))
    paths.update({HERE/'handle_macro_entry.py',HERE/'handle_macro_runtime.py',HERE/'handle_macro_route.py',HERE/'handle_full_runtime.py',HERE/'handle_full_route_plan.py',HERE/'handle_local_candidate_v4.py',HERE/'handle_local_driver_v4r4.py',HERE/'rgbd_capture_retry.py',route,Path(selected_runtime['robot_pad_geometry_file']),REPO/'tests/test_handle_macro_runtime.py'})
    for path in paths:bound[str(path.resolve())]=digest(path)
    return report,effective,bundle,bound


def guard(scope_path,freeze_path,bound,*,mode,output,runtime_sha):
    if not scope_path or not freeze_path:raise SafetyError('result-free shared pipeline scope/freeze required before start')
    scope=json.loads(scope_path.read_text());freeze=json.loads(freeze_path.read_text())
    if scope.get('execution_authorized') is not True or scope.get('task')!=TASK or scope.get('mode','sim')!=mode:
        raise SafetyError('scope does not authorize this exact pipeline backend')
    if mode=='sim' and (scope.get('maximum_full_simulator_episodes',0)<1 or scope.get('hardware_motion_jobs')!=0):raise SafetyError('simulator-only scope missing episode budget')
    if freeze.get('schema')!='graph-self-adapter.shared-cup-freeze.v1' or freeze.get('status')!='result_free_frozen' or freeze.get('scope_sha256')!=digest(scope_path):
        raise SafetyError('shared pipeline freeze mismatch')
    if freeze.get('runtime_sha256')!=runtime_sha:raise SafetyError('sim/real shared RuntimeHash mismatch')
    actual={str(Path(r['path']).resolve()):r['sha256'] for r in freeze['bindings']}
    if len(actual)!=len(freeze['bindings']):raise SafetyError('duplicate shared source binding')
    required={**bound,str(scope_path.resolve()):digest(scope_path)}
    if not required.items()<=actual.items() or any(digest(p)!=sha for p,sha in actual.items()):raise SafetyError('shared source/graph/model/SAM/scene binding changed')
    if freeze.get('jobs')!=[{'task':TASK,'mode':mode,'output':str(output.resolve())}]:raise SafetyError('exact shared pipeline job absent')
    reservation=Path(freeze.get('formal_reservation',''))
    if not reservation.is_file():raise SafetyError('formal macro reservation absent')
    claim=json.loads(reservation.read_text());ledger=json.loads(Path(claim['ledger_path']).read_text())
    if claim.get('scope_sha256')!=digest(scope_path) or claim.get('output')!=str(output.resolve()) or claim not in ledger.get('runs',[]) or len(ledger['runs'])>scope.get('maximum_full_simulator_episodes',0):raise SafetyError('formal parent scope run budget/reservation invalid')
    if mode=='real':
        if scope.get('hardware_motion_commands_authorized') is not True:raise SafetyError('hardware motion not authorized')
        gate=scope.get('passed_simulation_report')
        if not gate or digest(gate['path'])!=gate['sha256']:raise SafetyError('passing same-runtime SAM simulation required before hardware')
        sim=json.loads(Path(gate['path']).read_text())
        if not sim.get('strict_success') or sim.get('shared_runtime_sha256')!=runtime_sha or sim.get('mask_source')!='actual_SAM2':
            raise SafetyError('same-runtime full SAM simulation gate has not passed')


def runtime_hash():
    import hashlib
    paths=(HERE/'handle_macro_entry.py',HERE/'handle_macro_runtime.py',HERE/'handle_macro_route.py',HERE/'handle_full_runtime.py',HERE/'shared_cup_runtime.py',HERE/'object_centered_runtime.py',REPO/'prove/xtrainer/runtime.py',REPO/'prove/xtrainer/agent.py',REPO/'prove/xtrainer/perception.py',HERE/'handle_local_candidate_v4.py',HERE/'handle_local_driver_v4r4.py',HERE/'handle_full_route_plan.py',HERE/'rgbd_capture_retry.py')
    return hashlib.sha256(''.join(digest(p) for p in paths).encode()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('sim',),nargs='?',default='sim')
    p.add_argument('--config',type=Path,default=REPO/'task_packages/cup_sam_closed_loop_v01/CONFIG_MACRO.json')
    p.add_argument('--scope',type=Path);p.add_argument('--execution-freeze',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--check',action='store_true')
    a=p.parse_args(argv);cfg=json.loads(a.config.read_text());report,runtime_cfg,bundle,bound=bindings(a.config,cfg,a.mode)
    runtime_source=HERE/'handle_macro_runtime.py'
    runtime_sha=runtime_hash()
    ready={'variant':VARIANT,'mode':a.mode,'shared_runtime_sha256':runtime_sha,'same_graph_model_Code_NN_Root_interface':True,
           'mask_source':'actual_SAM2','GT_segmentation_fallback':False,'hardware_connections':0,
           'required_bindings':[{'path':p,'sha256':sha} for p,sha in sorted(bound.items())],'dependencies':report['dependencies']}
    if a.check:print(json.dumps(ready,indent=2));return ready
    if not a.output or a.output.exists():raise SafetyError('fresh explicit output required')
    guard(a.scope,a.execution_freeze,bound,mode=a.mode,output=a.output,runtime_sha=runtime_sha)
    runtime=load(runtime_source,'_shared_ObjectCenteredExecutor')
    rolevision=runtime
    if a.mode=='real':
        # One explicit backend choice declares this motion job; no invented physical review flags.
        from prove.xtrainer.io import NativeRPC,DashboardStop,SafetyLayer,feedback_fk_check
        from prove.xtrainer.perception import ZEDCamera
        framed=load(HERE/'real_feedback_stream.py','_shared_direct_feedback')
        streams=[framed.FramedFeedbackStream(runtime_cfg['robot_ips'][arm],runtime_cfg['sdk_file']) for arm in ('left','right')]
        stops=[DashboardStop(runtime_cfg['robot_ips'][arm]) for arm in ('left','right')];rpc=NativeRPC()
        import numpy as np
        warm=load(HERE/'operator_launcher.py','_shared_direct_startup').startup_vision_class(rolevision.SharedSAMVision,SafetyError)
        vision=warm(ZEDCamera(),np.asarray(runtime_cfg['T_left_from_camera']),runtime_cfg['vision'],evidence_root=a.output/'SAM_MASK_EVIDENCE')
        class RealRobot:
            @property
            def commands(self):return rpc.commands
            def samples(self):return [stream.snapshot() for stream in streams]
            def fk(self,*args):return rpc.fk(*args)
            def command(self,*args):return rpc.command(*args)
        try:
            for stream in streams:stream.connect()
            for stop in stops:stop.connect()
            rpc.connect();samples=[stream.snapshot() for stream in streams];SafetyLayer(runtime_cfg['safety']).samples(samples)
            comparison=feedback_fk_check(rpc,samples,runtime_cfg['known_initial_gripper_commands'])
            if not comparison['passed']:raise SafetyError('actual native TCP/FK consistency failure')
            vision.connect();runtime_cfg['mode']='execute'
            judge=RealRootJudge(a.output/'REAL_AGENT_HANDOFFS',task=TASK,graph_sha=digest(bundle['paths']['graph']),identity=runtime_cfg['root_identity'])
            result=runtime.SharedCupExecutor(bundle,RealRobot(),vision,judge,stops,runtime_cfg,a.output).run()
            result.update(shared_runtime_sha256=runtime_sha,mask_source='actual_SAM2',variant=VARIANT,operator_motion_declaration='explicit real mode',
                          no_physical_qualification_inferred=True,native_feedback_FK_check=comparison)
            write(a.output/'RESULT.json',result);return result
        finally:
            rpc.close();vision.close()
            for stream in streams:stream.close()
            for stop in stops:stop.close()
    # This is a formal scope-frozen simulation execution, not a dry check.
    os.environ.setdefault('MUJOCO_GL','egl');simio=load(HERE/'sim_robot_io.py','_shared_sim_robot_io')
    import numpy as np
    from prove.xtrainer.perception import LiveVision
    warm=load(HERE/'operator_launcher.py','_shared_startup_vision').startup_vision_class(rolevision.SharedSAMVision,SafetyError)
    backend=simio.SimRobot(scene_run=cfg['sim']['scene_run'],physics_source=cfg['sim']['physics_source'],bundle=bundle,
        T_left_from_right=runtime_cfg['T_left_from_right'],initial_q=cfg['sim'].get('initial_robot_q_rad'))
    camera=simio.SimRGBDCamera(backend);vision=None
    runtime_cfg.update(mode='sim',known_initial_gripper_commands=bundle['initial_g'].tolist())
    # Simulation encoder initialization is declared; no operator confirmation Boolean is fabricated.
    judge=RealRootJudge(a.output/'REAL_AGENT_HANDOFFS',task=TASK,graph_sha=digest(bundle['paths']['graph']),identity=runtime_cfg['root_identity'])
    try:
        camera.connect();backend.start()
        vision=warm(camera,camera.T,runtime_cfg['vision'],evidence_root=a.output/'SAM_MASK_EVIDENCE');vision.connect=lambda:None
        stops=[simio.SimStop(backend,arm) for arm in ('left','right')]
        executor=runtime.SharedCupExecutor(bundle,backend,vision,judge,stops,runtime_cfg,a.output)
        vision.stage_context=lambda:executor.stage
        result=executor.run()
        result.update(shared_runtime_sha256=runtime_sha,mask_source='actual_SAM2',simulation_command_count=backend.commands,hardware_motion_commands=0,
            simulation_motion_command_attempts=result.get('hardware_command_attempts',0),
            simulation_motion_command_acknowledgments=result.get('hardware_command_acknowledgments',0),
            hardware_command_attempts=0,hardware_command_acknowledgments=0,simulation_encoder_initialization_is_not_hardware_confirmation=True)
        write(a.output/'PREDICTIONS_SEALED.json',{'predictions':result,'Root_records':judge.records,'physics_integrity':backend.integrity(),
                                              'sealed_before_private_evaluator':True,'shared_runtime_sha256':runtime_sha})
        # End integration only after the episode, never while Root reviews.
        backend.stop_event.set();backend.thread.join(timeout=3)
        # Private full state is opened ONLY after prediction sealing and episode end.
        private_trace=backend.world.evaluator_trace()
        state_path=a.output/'STATE_TRACE.npz'
        np.savez_compressed(state_path,timestamps=np.asarray([r['physical_time_sec'] for r in private_trace]),
            qpos=np.asarray([r['qpos'] for r in private_trace]),ctrl=np.asarray([r['ctrl'] for r in private_trace]))
        result['post_seal_state_trace']={'path':str(state_path),'sha256':digest(state_path),'evaluator_only_not_runtime_input':True}
        external=Path(cfg['sim']['evaluator_root']);sys.path.insert(0,str(external));from twin.physics_evaluation import task_outcome
        outcome=backend.world.finalize_evaluator(task_outcome)
        result.update(variant=VARIANT,mask_source='actual_SAM2',shared_runtime_sha256=runtime_sha,simulation_physics_continues_during_Root=True,
            physics_integrity=backend.integrity(),post_seal_private_evaluation=outcome,
            strict_success=bool(result['graph_completed'] and outcome['task_outcome'].get('passed') and outcome['numerical_safety'].get('passed')))
        write(a.output/'RESULT.json',result);return result
    finally:
        if vision:vision.close()
        elif camera.renderer:camera.close()
        backend.close()

if __name__=='__main__':main()
