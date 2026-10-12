#!/usr/bin/env python3
"""Task-bound prepared real adapter. Default check: no key, device or API."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (HERE,REPO):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from prove.xtrainer.agent import digest,write
from prove.xtrainer.io import SafetyError,load_calibration
from ordinary_real_portable_v21 import load_bundle,check


def prepare(root,task,config):
    b,report=load_bundle(root,task);manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text());cfg=copy.deepcopy(json.loads(config.read_text()))
    from multitask_semantic_judge import validate_config
    validate_config(cfg['stage_judge']) # source/model contract only: never opens key
    missing=[]
    resources={'sdk_file':cfg['sdk_file'],'left_calibration':cfg['left_calibration'],'right_calibration':cfg['right_calibration'],'sam_utility':cfg['vision']['sam_utility'],'sam_checkpoint':cfg['vision']['sam_checkpoint']}
    for key,path in resources.items():
        if not Path(path).is_file():missing.append(key)
        elif digest(path)!=manifest['external_resource_sha256'][key]:raise SafetyError('real external resource hash mismatch: '+key)
    if not {'left_calibration','right_calibration'}&set(missing):
        camera,right=load_calibration(cfg['left_calibration'],cfg['right_calibration']);cfg.update(T_left_from_camera=camera.tolist(),T_left_from_right=right.tolist())
    report.update(missing_external_resources=missing,initial_gripper_command_is_not_measurement=True,initial_Grounding='current Root fresh RGB role boxes',same_runtime_modern_pass_required=True)
    return b,cfg,report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('mode',choices=('check','shadow','execute'),nargs='?',default='check');p.add_argument('--package',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);a=p.parse_args(argv)
    root=a.package.resolve();config=a.config or root/'local/CONFIG_REAL.json';b,cfg,report=prepare(root,a.task,config)
    if a.mode=='check':print(json.dumps(report,indent=2));return report
    if not report['promotion_verified']:raise SafetyError('physical I/O held: same-runtime complete modern simulation receipt not present')
    if report['missing_external_resources']:raise SafetyError('physical I/O held: required resource paths missing')
    scope=root/'local/SCOPE_REAL.json';authority=json.loads(scope.read_text())
    if authority.get('task')!=a.task or authority.get('read_only_shadow_authorized' if a.mode=='shadow' else 'hardware_motion_commands_authorized') is not True:raise SafetyError('existing task motion/read-only intent absent')
    output=a.output or root/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+a.mode)
    if output.exists():raise SafetyError('fresh real output required')
    output.mkdir(parents=True);cfg['mode']='execute';cfg['vision'].update(role_boxes={},prompt_frame_sha256=None,prompt_frame_output=str(output/'INITIAL_RGB.png'),prompt_reply_path=str(output/'INITIAL_BOXES.json'))
    write(output/'EFFECTIVE_CONFIG.json',cfg);manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text());bound={str(root/r['path']):r['sha256'] for r in manifest['files']}
    for path in (root/'PACKAGE_MANIFEST.json',root/'PROMOTION_RECEIPT.json',config,scope,output/'EFFECTIVE_CONFIG.json'):bound[str(path.resolve())]=digest(path)
    write(output/'EXECUTION_FREEZE.json',{'schema':'graph-self-adapter.ordinary-real-freeze.v21','status':'result_free_frozen','task':a.task,'mode':a.mode,'runtime_sha256':report['runtime_sha256'],'bindings':[{'path':path,'sha256':sha} for path,sha in sorted(bound.items())],'operator_intent':'explicit execute, existing session task authorization','no_physical_qualification_inferred':True})
    if any(digest(path)!=sha for path,sha in bound.items()):raise SafetyError('real source/config changed before connection')
    from prove.xtrainer.io import NativeRPC,DashboardStop,SafetyLayer,feedback_fk_check
    from prove.xtrainer.perception import ZEDCamera
    from real_feedback_stream import FramedFeedbackStream
    from operator_launcher import startup_vision_class
    from multitask_semantic_judge import MultitaskSemanticJudge
    from multitask_ordinary_runtime_v21 import RelativeLiftExecutor,SharedSAMVision
    rpc=NativeRPC();streams=[FramedFeedbackStream(cfg['robot_ips'][arm],cfg['sdk_file']) for arm in ('left','right')];stops=[DashboardStop(cfg['robot_ips'][arm]) for arm in ('left','right')] if a.mode=='execute' else []
    warm=startup_vision_class(SharedSAMVision,SafetyError);vision=warm(ZEDCamera(),np.asarray(cfg['T_left_from_camera']),cfg['vision'],evidence_root=output/'SAM_MASK_EVIDENCE')
    class Robot:
        @property
        def commands(self):return rpc.commands
        def samples(self):return [stream.snapshot() for stream in streams]
        def fk(self,*args):return rpc.fk(*args)
        def command(self,*args):
            if a.mode!='execute':raise SafetyError('shadow command forbidden')
            return rpc.command(*args)
    try:
        for stream in streams:stream.connect()
        for stop in stops:stop.connect()
        rpc.connect();robot=Robot();samples=robot.samples();SafetyLayer(cfg['safety']).samples(samples)
        vision.connect()
        if a.mode=='shadow':
            # Read-only own FK/actual RGBD. No task motion/model API or key read.
            observation=vision.capture();samples=robot.samples();SafetyLayer(cfg['safety']).samples(samples);start=time.monotonic();comparison=feedback_fk_check(robot,samples,cfg['known_initial_gripper_commands'])
            age=time.monotonic()-observation.stamp
            result={'task':a.task,'mode':'shadow','hardware_commands':0,'api_calls':0,'actual_aperture_m':[s.aperture for s in samples],'roles':{k:None if v is None else v.tolist() for k,v in observation.roles.items()},'FK_feedback':comparison,'FK_check_wall_sec':time.monotonic()-start,'RGBD_age_sec':age,'passed':bool(comparison['passed'] and 0<=age<=cfg['safety']['max_age_sec'])}
        else:
            comparison=feedback_fk_check(robot,samples,cfg['known_initial_gripper_commands'])
            if not comparison['passed']:raise SafetyError('actual own FK/feedback mismatch')
            judge=MultitaskSemanticJudge(output/'REAL_AGENT_HANDOFFS',task=a.task,graph_sha=digest(b['paths']['graph']),key_file=cfg['stage_judge']['key_file'],maximum_calls=cfg['stage_judge']['maximum_calls'])
            result=RelativeLiftExecutor(b,robot,vision,judge,stops,cfg,output).run();result.update(task=a.task,runtime_sha256=report['runtime_sha256'],judge_identity=judge.identity,API_records=judge.api_records,physical_qualification_inferred=False)
        write(output/'RESULT.json',result);return result
    except Exception as error:
        write(output/'FAILURE.json',{'task':a.task,'mode':a.mode,'failure':str(error),'acknowledged_commands':rpc.commands,'ACK_failure_does_not_mean_no_motion':True});raise
    finally:
        rpc.close();vision.close()
        for stream in streams:stream.close()
        for stop in stops:stop.close()

if __name__=='__main__':main()
