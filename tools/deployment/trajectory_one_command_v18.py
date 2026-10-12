#!/usr/bin/env python3
"""One-command formal simulator run. Scope quota is shared across invocations."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from prove.xtrainer.agent import digest,write
from prove.xtrainer.io import SafetyError
import trajectory_entry_v18 as entry
from direct_white_route import direct_route_graph
from macro_one_command import reserve

OWNER=REPO/'evaluations/stage_trajectory_v18'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('sim',),nargs='?',default='sim')
    p.add_argument('--config',type=Path,default=OWNER/'CONFIG_PROPOSED.json');p.add_argument('--scope',type=Path,default=OWNER/'TRIAL_SCOPE.json')
    p.add_argument('--check',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    config_source=a.config.resolve();scope=a.scope.resolve();cfg=copy.deepcopy(json.loads(config_source.read_text()))
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8];output=OWNER/'runs'/stamp;control=OWNER/'control'/stamp
    graph=json.loads((Path(cfg['portable_package'])/'graph/task_graph.json').read_text());routed=direct_route_graph(graph)
    cfg.pop('qualified_handle_candidate',None);cfg.pop('handle_route_graph',None)
    cfg['direct_route_graph']=str(control/'TASK_GRAPH.json');cfg['formal_run_reservation']=str(control/'FORMAL_RUN_RESERVATION.json')
    cfg['runtime']['vision'].update(prompt_frame_output=str(output/'INITIAL_RGB.png'),prompt_reply_path=str(output/'INITIAL_BOXES.json'),prompt_frame_sha256=None,role_boxes={})
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    if a.check:
        from prove.xtrainer.portable import portable_check
        report,_,_=portable_check(Path(cfg['portable_package']),cfg['runtime'])
        import OpenGL.EGL as egl
        if not hasattr(egl,'EGLDeviceEXT'):raise SafetyError('EGL rendering dependency missing')
        declared=json.loads(scope.read_text())
        if declared.get('execution_authorized') is not True or declared.get('hardware_motion_jobs')!=0:raise SafetyError('formal scope absent')
        result={'mode':'sim','scope':str(scope),'config_source':str(config_source),'planned_fresh_output':str(output),'macro_route_stages':len(routed['stages']),
            'runtime_sha256':entry.runtime_hash(),'hardware_connections':0,'episode_started':False,'ledger_changed':False,'package_bindings_verified':True,'model_graph_reference_hashes_verified':True,'EGLDeviceEXT_available':True,'scope_budget':declared['maximum_full_simulator_episodes']}
        print(json.dumps(result,indent=2));return result
    if control.exists() or output.exists():raise SafetyError('fresh formal output/control required')
    control.mkdir(parents=True);write(control/'TASK_GRAPH.json',routed)
    reservation=reserve(scope,OWNER/'FORMAL_RUN_LEDGER.json',output,control);write(control/'FORMAL_RUN_RESERVATION.json',reservation)
    config=control/'CONFIG.json';write(config,cfg);_,_,_,bound=entry.bindings(config,cfg,'sim')
    extra={str(config_source):digest(config_source),str(scope):digest(scope),str(Path(__file__).resolve()):digest(__file__),str(REPO/'task_packages/cup_sam_closed_loop_v01/run_trajectory_v18.sh'):digest(REPO/'task_packages/cup_sam_closed_loop_v01/run_trajectory_v18.sh')}
    bound.update(extra);freeze={'schema':'graph-self-adapter.shared-cup-freeze.v1','status':'result_free_frozen','scope_sha256':digest(scope),'runtime_sha256':entry.runtime_hash(),
        'bindings':[{'path':p,'sha256':sha} for p,sha in sorted(bound.items())],'jobs':[{'task':entry.TASK,'mode':'sim','output':str(output)}],
        'formal_reservation':str(control/'FORMAL_RUN_RESERVATION.json'),'scope_not_cloned_or_reset':True}
    freeze_path=control/'EXECUTION_FREEZE.json';write(freeze_path,freeze)
    entry.guard(scope,freeze_path,bound,mode='sim',output=output,runtime_sha=entry.runtime_hash())
    result={'output':str(output),'config':str(config),'scope':str(scope),'freeze':str(freeze_path),'formal_run_index':reservation['run_index'],'Root_grounding_and_boundary_judgement_required':True}
    print(json.dumps(result,indent=2),flush=True)
    if a.prepare_only:return result
    return entry.main(['sim','--config',str(config),'--scope',str(scope),'--execution-freeze',str(freeze_path),'--output',str(output)])

if __name__=='__main__':main()
