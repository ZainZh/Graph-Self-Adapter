#!/usr/bin/env python3
"""Per-task fresh output/config/freeze; one scoped simulator episode per invocation."""
import argparse,copy,json,os,sys,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (HERE,REPO):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from prove.xtrainer.agent import digest,write
from prove.xtrainer.io import SafetyError
from prove.xtrainer.multitask_bundle import load_bundle
from multitask_ordinary_route import ordinary_route
import multitask_ordinary_entry_v21 as entry
from macro_one_command import reserve


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--owner-root',type=Path,default=REPO/'evaluations/multitask_runtime_v21');p.add_argument('mode',choices=('check','sim','shadow','real'),nargs='?',default='check');p.add_argument('--config',type=Path);p.add_argument('--scope',type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    owner=a.owner_root.resolve()/a.task;config_source=a.config or owner/'CONFIG_PROPOSED.json';scope=a.scope or owner/'TRIAL_SCOPE.json'
    ready=entry.bundle_check(a.task)
    if a.mode=='check':ready.update(configured=config_source.is_file(),formal_scope_available=scope.is_file(),source_config=str(config_source),task_selected_explicitly=True);print(json.dumps(ready,indent=2));return ready
    if not config_source.is_file() or not scope.is_file():raise SafetyError('task-specific configuration and validation/promotion scope missing; no cup default')
    cfg=copy.deepcopy(json.loads(config_source.read_text()))
    if cfg.get('task')!=a.task:raise SafetyError('per-task config identity mismatch')
    if a.mode!='sim':raise SafetyError('same-runtime full task simulation receipt and portable real activation still required; no old GT or cup fallback')
    authority=json.loads(scope.read_text())
    if authority.get('task')!=a.task:raise SafetyError('exact task-scoped simulator authority required')
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8];output=owner/'runs'/stamp;control=owner/'control'/stamp
    if output.exists() or control.exists():raise SafetyError('fresh per-task output required')
    control.mkdir(parents=True);route=control/'TASK_GRAPH.json';write(route,ordinary_route(load_bundle(a.task)['graph']))
    claim=reserve(scope,owner/'FORMAL_RUN_LEDGER.json',output,control);write(control/'FORMAL_RUN_RESERVATION.json',claim)
    cfg.update(ordinary_route_graph=str(route),formal_run_reservation=str(control/'FORMAL_RUN_RESERVATION.json'))
    cfg['runtime']['vision'].update(prompt_frame_output=str(output/'INITIAL_RGB.png'),prompt_reply_path=str(output/'INITIAL_BOXES.json'),prompt_frame_sha256=None,role_boxes={})
    config=control/'CONFIG.json';write(config,cfg);_,_,_,bound=entry.bindings(config,cfg,'sim')
    for path in (config_source,scope,Path(__file__),HERE/'macro_one_command.py',REPO/'task_packages/multitask_v21'/a.task/'run.sh'):bound[str(path.resolve())]=digest(path)
    freeze={'schema':'graph-self-adapter.shared-cup-freeze.v1','status':'result_free_frozen','scope_sha256':digest(scope),'runtime_sha256':entry.runtime_hash(),'bindings':[{'path':path,'sha256':sha} for path,sha in sorted(bound.items())],'jobs':[{'task':a.task,'mode':'sim','output':str(output)}],'formal_reservation':str(control/'FORMAL_RUN_RESERVATION.json'),'scope_not_cloned_or_reset':True,'generic_task':a.task}
    freeze_path=control/'EXECUTION_FREEZE.json';write(freeze_path,freeze);entry.backend.TASK=a.task;entry.backend.guard(scope,freeze_path,bound,mode='sim',output=output,runtime_sha=entry.runtime_hash())
    print(json.dumps({'task':a.task,'output':str(output),'freeze':str(freeze_path),'required_initial_SAM_roles':ready['roles'],'semantic_budget':ready['required_supported_judgements']},indent=2),flush=True)
    if a.prepare_only:return freeze
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    return entry.main(['--task',a.task,'sim','--config',str(config),'--scope',str(scope),'--execution-freeze',str(freeze_path),'--output',str(output)])

if __name__=='__main__':main()
