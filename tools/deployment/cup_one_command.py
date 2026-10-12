#!/usr/bin/env python3
"""Independent one-command preparation; never imported by an active episode.

Configured Root scope remains the authority. real additionally requires strict
same-RuntimeHash actual-SAM simulator success; preparation grants no permission.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid


def load_entry(path):
    spec=importlib.util.spec_from_file_location('_one_command_shared_entry',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def resolve(base,value):
    path=Path(value).expanduser();return (base/path).resolve() if not path.is_absolute() else path.resolve()


def prepare(defaults_path,mode):
    defaults_path=Path(defaults_path).resolve();defaults=json.loads(defaults_path.read_text());base=defaults_path.parent
    if defaults.get('schema')!='graph-self-adapter.cup-one-command-defaults.v1':raise ValueError('Root-created defaults schema required')
    profile=defaults['profiles'][mode]
    entry=load_entry(resolve(base,defaults['entry_source']));config=resolve(base,profile['config']);cfg=json.loads(config.read_text())
    _,_,_,bound=entry.bindings(config,cfg,mode)
    import hashlib
    runtime_files=(entry.HERE/'shared_cup_runtime.py',entry.HERE/'object_centered_runtime.py',entry.REPO/'prove/xtrainer/runtime.py',
                   entry.REPO/'prove/xtrainer/agent.py',entry.REPO/'prove/xtrainer/perception.py',entry.HERE/'shared_cup_vision.py')
    runtime_sha=hashlib.sha256(''.join(entry.digest(p) for p in runtime_files).encode()).hexdigest()
    scope_template=resolve(base,profile['scope']);scope=copy.deepcopy(json.loads(scope_template.read_text()))
    if scope.get('execution_authorized') is not True or scope.get('task')!=entry.TASK or scope.get('mode','sim')!=mode:
        raise ValueError('configured scope does not authorize selected backend')
    if mode=='sim' and scope.get('operator_reuse_authorized') is not True:
        raise ValueError('one-command sim requires a separate Root-authorized operator-reuse scope')
    # No automatic Boolean change and no private hardware qualification claim.
    if mode=='real':
        if scope.get('hardware_motion_commands_authorized') is not True:raise ValueError('configured hardware motion authority absent')
        passed=scope.get('passed_simulation_report')
        if not passed or entry.digest(passed['path'])!=passed['sha256']:raise ValueError('strict same-runtime SAM simulation source absent or changed')
        evidence=json.loads(Path(passed['path']).read_text())
        if evidence.get('strict_success') is not True or evidence.get('shared_runtime_sha256')!=runtime_sha or evidence.get('mask_source')!='actual_SAM2':
            raise ValueError('strict same-runtime SAM simulation gate not passed')
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    output=resolve(base,profile['output_root'])/stamp;control=resolve(base,profile['control_root'])/stamp
    if output.exists() or control.exists():raise FileExistsError('fresh timestamp paths required')
    return entry,config,scope_template,scope,bound,runtime_sha,output,control


def fresh_config_plan(config, mode, control, output):
    cfg=copy.deepcopy(json.loads(Path(config).read_text()))
    auxiliary={}
    if mode=='real':
        original=Path(cfg['real_operator_config']).resolve()
        runtime=copy.deepcopy(json.loads(original.read_text()))
        new_runtime=control/'REAL_RUNTIME_CONFIG.json'
        cfg['real_operator_config']=str(new_runtime)
        auxiliary[new_runtime]=runtime
    else:runtime=cfg['runtime']
    vision=runtime['vision']
    vision.update(prompt_frame_output=str(output/'GROUNDING/current_first_frame.png'),
                  prompt_reply_path=str(output/'GROUNDING/current_first_frame_boxes.json'),
                  prompt_frame_sha256=None,role_boxes={})
    return cfg,auxiliary


def rendering_dependency_check():
    # Import verification only, not a graphics context, camera or simulator world.
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import OpenGL.EGL as egl
    if not hasattr(egl,'EGLDeviceEXT'):raise RuntimeError('PyOpenGL EGLDeviceEXT unavailable for MuJoCo EGL rendering')
    import mujoco
    return {'mujoco':mujoco.__version__,'EGLDeviceEXT_available':True,'graphics_context_created':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('sim','real'))
    p.add_argument('--defaults',type=Path,default=Path(__file__).resolve().parents[2]/'task_packages/cup_sam_closed_loop_v01/DEFAULTS.json')
    p.add_argument('--check',action='store_true');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv);entry,config,source_scope,scope,bound,runtime_sha,output,control=prepare(a.defaults,a.mode)
    rendering=rendering_dependency_check() if a.mode=='sim' else None
    if a.check:
        result={'mode':a.mode,'config':str(config),'scope':str(source_scope),'future_output':str(output),
                'shared_runtime_sha256':runtime_sha,'hardware_connections':0,'episode_started':False,'render_dependencies':rendering}
        print(json.dumps(result,indent=2));return result
    control.mkdir(parents=True,exist_ok=False)
    source_config=config
    cfg,auxiliary=fresh_config_plan(source_config,a.mode,control,output)
    for path,value in auxiliary.items():entry.write(path,value)
    config=control/'CONFIG.json';entry.write(config,cfg)
    _,_,_,bound=entry.bindings(config,cfg,a.mode)
    bound[str(source_config)]=entry.digest(source_config)
    if a.mode=='real':
        original_runtime=Path(json.loads(source_config.read_text())['real_operator_config']).resolve()
        bound[str(original_runtime)]=entry.digest(original_runtime)
    generated_scope=control/'SCOPE.json';entry.write(generated_scope,scope)
    bindings={**bound,str(generated_scope):entry.digest(generated_scope),str(source_scope):entry.digest(source_scope),
              str(Path(a.defaults).resolve()):entry.digest(a.defaults),str(Path(__file__).resolve()):entry.digest(__file__)}
    freeze={'schema':'graph-self-adapter.shared-cup-freeze.v1','status':'result_free_frozen','scope_sha256':entry.digest(generated_scope),
            'runtime_sha256':runtime_sha,'bindings':[{'path':p,'sha256':sha} for p,sha in sorted(bindings.items())],
            'jobs':[{'task':entry.TASK,'mode':a.mode,'output':str(output)}],
            'preparation_is_not_new_authorization':True,'scope_authority_source':str(source_scope)}
    freeze_path=control/'EXECUTION_FREEZE.json';entry.write(freeze_path,freeze)
    entry.guard(generated_scope,freeze_path,bound,mode=a.mode,output=output,runtime_sha=runtime_sha)
    result={'mode':a.mode,'config':str(config),'scope':str(generated_scope),'freeze':str(freeze_path),'output':str(output),
            'manual_current_Root_node_image_judgement_required':True,'unattended_service':False}
    print(json.dumps(result,indent=2))
    if a.prepare_only:return result
    return entry.main([a.mode,'--config',str(config),'--scope',str(generated_scope),'--execution-freeze',str(freeze_path),'--output',str(output)])

if __name__=='__main__':main()
