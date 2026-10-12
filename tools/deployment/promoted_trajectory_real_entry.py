#!/usr/bin/env python3
"""Promoted V18 trajectory runtime with thin real I/O; default is file-only check."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]))
from trajectory_promotion_check import check
from prove.xtrainer.agent import digest,write,RealRootJudge
from prove.xtrainer.io import SafetyError


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def prepare(package_root,config_path,receipt_sha):
    from prove.xtrainer.portable import portable_check
    root=Path(package_root).resolve();cfg=json.loads(Path(config_path).read_text());runtime=copy.deepcopy(cfg.get('runtime',cfg))
    resources={'sdk_file':runtime['sdk_file'],'left_calibration':runtime['left_calibration'],'right_calibration':runtime['right_calibration'],
               'sam_checkpoint':runtime['vision']['sam_checkpoint'],'sam_utility':runtime['vision']['sam_utility']}
    report=check(root,receipt_sha,external_paths=resources)
    if runtime.get('robot_TCP_offset_m')!=.197:raise SafetyError('explicit real Tool1 TCP197mm required')
    runtime['robot_pad_geometry_file']=str(root/'data/ROBOT_FLAT_PAD_GEOMETRY.json')
    prior=root/'data/SAM_STATIC_REFERENCE.json'
    if prior.exists():runtime['static_reference_points_file']=str(prior)
    portable_report,effective,bundle=portable_check(root/'package',runtime)
    bundle['graph']=json.loads((root/'data/TASK_GRAPH.json').read_text());bundle['paths']['graph']=str(root/'data/TASK_GRAPH.json')
    if len(bundle['graph']['stages'])!=json.loads((root/'PROMOTION_RECEIPT.json').read_text())['graph_nodes']:raise SafetyError('promoted graph count differs')
    return report,effective,bundle


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('check','shadow','execute'),nargs='?',default='check')
    p.add_argument('--package',type=Path,default=HERE.parents[2]);p.add_argument('--config',type=Path);p.add_argument('--scope',type=Path)
    p.add_argument('--receipt-sha256');p.add_argument('--output',type=Path);a=p.parse_args(argv)
    root=a.package.resolve();config=a.config or root/'local/CONFIG_REAL.json';scope=a.scope or root/'local/SCOPE_REAL.json'
    approved=a.receipt_sha256 or (root/'local/ROOT_RECEIPT_SHA256.txt').read_text().strip()
    report,cfg,bundle=prepare(root,config,approved)
    if a.mode=='check':
        print(json.dumps({**report,'mode':'check','package_inputs_verified':True,'graph_nodes':len(bundle['graph']['stages']),'device_connections':0,'hardware_commands':0},indent=2));return report
    authority=json.loads(scope.read_text())
    if authority.get('task')!='put_mug_on_box' or (a.mode=='execute' and authority.get('hardware_motion_commands_authorized') is not True) or (a.mode=='shadow' and authority.get('read_only_shadow_authorized') is not True):raise SafetyError('selected real operation not authorized by independent scope')
    output=a.output or root/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+a.mode)
    if output.exists():raise SafetyError('fresh real output required')
    output.mkdir(parents=True);cfg['mode']='execute';cfg['vision'].update(role_boxes={},prompt_frame_sha256=None,prompt_frame_output=str(output/'INITIAL_RGB.png'),prompt_reply_path=str(output/'INITIAL_BOXES.json'))
    write(output/'EFFECTIVE_CONFIG.json',cfg)
    receipt=json.loads((root/'PROMOTION_RECEIPT.json').read_text());bound={str(root/r['relative_path']):r['sha256'] for r in receipt['files']}
    bound.update({str(Path(__file__).resolve()):digest(__file__),str(config.resolve()):digest(config),str(scope.resolve()):digest(scope),str(output/'EFFECTIVE_CONFIG.json'):digest(output/'EFFECTIVE_CONFIG.json')})
    freeze={'schema':'graph-self-adapter.promoted-real-freeze.v1','status':'result_free_frozen','shared_runtime_sha256':receipt['shared_runtime_sha256'],'receipt_sha256':approved,
        'bindings':[{'path':p,'sha256':sha} for p,sha in sorted(bound.items())],'job':{'mode':a.mode,'task':'put_mug_on_box','output':str(output)},
        'operator_motion_declaration':'explicit execute mode; existing user-authorized task scope','physical_review_or_jaw_measurement_inferred':False}
    write(output/'EXECUTION_FREEZE.json',freeze)
    if any(digest(path)!=sha for path,sha in bound.items()):raise SafetyError('real prepared binding changed')
    print(json.dumps({'output':str(output),'mode':a.mode,'shared_runtime_sha256':receipt['shared_runtime_sha256']},indent=2),flush=True)
    from prove.xtrainer.io import NativeRPC,DashboardStop,SafetyLayer,feedback_fk_check
    from prove.xtrainer.perception import ZEDCamera
    from real_feedback_stream import FramedFeedbackStream
    from operator_launcher import startup_vision_class
    import trajectory_white_runtime_v18 as runtime
    import numpy as np
    streams=[FramedFeedbackStream(cfg['robot_ips'][arm],cfg['sdk_file']) for arm in ('left','right')]
    stops=[DashboardStop(cfg['robot_ips'][arm]) for arm in ('left','right')] if a.mode=='execute' else []
    rpc=NativeRPC();warm=startup_vision_class(runtime.SharedSAMVision,SafetyError)
    vision=warm(ZEDCamera(serial=cfg['vision'].get('serial',38968158)),np.asarray(cfg['T_left_from_camera']),cfg['vision'],evidence_root=output/'SAM_MASK_EVIDENCE')
    class Robot:
        @property
        def commands(self):return rpc.commands
        def samples(self):return [s.snapshot() for s in streams]
        def fk(self,*args):return rpc.fk(*args)
        def command(self,*args):
            if a.mode!='execute':raise SafetyError('read-only shadow cannot command')
            return rpc.command(*args)
    try:
        for stream in streams:stream.connect()
        for stop in stops:stop.connect()
        rpc.connect();samples=[s.snapshot() for s in streams];SafetyLayer(cfg['safety']).samples(samples)
        comparison=feedback_fk_check(rpc,samples,cfg['known_initial_gripper_commands'])
        if not comparison['passed']:raise SafetyError('actual native FK/feedback inconsistency')
        vision.connect()
        if a.mode=='shadow':
            observation=vision.capture();samples=Robot().samples();SafetyLayer(cfg['safety']).samples(samples)
            result={'mode':'shadow','hardware_commands':0,'FK_feedback':comparison,'roles':{k:None if v is None else np.asarray(v).tolist() for k,v in observation.roles.items()},'actual_aperture_m':[s.aperture for s in samples],'shared_runtime_sha256':receipt['shared_runtime_sha256']}
        else:
            judge=RealRootJudge(output/'REAL_AGENT_HANDOFFS',task='put_mug_on_box',graph_sha=digest(root/'data/TASK_GRAPH.json'),identity=cfg['root_identity'])
            result=runtime.SharedCupExecutor(bundle,Robot(),vision,judge,stops,cfg,output).run()
            result.update(shared_runtime_sha256=receipt['shared_runtime_sha256'],simulation_reexecuted=False,physical_qualification_inferred=False,native_FK_feedback=comparison)
        write(output/'RESULT.json',result);return result
    finally:
        rpc.close();vision.close()
        for stream in streams:stream.close()
        for stop in stops:stop.close()

if __name__=='__main__':main()
