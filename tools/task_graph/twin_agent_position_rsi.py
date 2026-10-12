#!/usr/bin/env python3
"""Additive cup-only Agent regression over the ten opened v02 position conditions.

Dry-run never installs physics; execute requires ten exact result-free jobs.
The immutable v06 pause/judge/hold core and original v02 perturbation are reused.
"""
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from tools.task_graph import twin_agent_live_rsi as core
from prove.task_graph.agent_stage_judge import digest
from prove.task_graph.agent_live_loop import HandoffError,atomic_json
from prove.task_graph.twin_sensor_judge_repair1 import RepairConfig

TASK='put_mug_on_box'
VARIANT='agent_live_position_v07_position_v01'
FREEZE_SCHEMA='graph-self-adapter.agent-position-freeze.v1'
OWNER=REPO/'evaluations/agent_stage_v07_position_v01'
RUN_ROOT=REPO/'runs/agent_stage_v07_position_v01'
CONDITIONS=REPO/'evaluations/twin_rsi_v02/FINAL_TEST_CONDITIONS.json'
SELECTION=REPO/'evaluations/twin_rsi_v02/FINAL_SELECTION.json'
HISTORY=REPO/'evaluations/agent_stage_v06/EXECUTION_FREEZE.json'
FACT_IDS=['semantic.continue_placing_permission','semantic.align_ready','semantic.placement_complete']


def checked_paths(freeze):
    paths=set()
    for row in freeze['bindings']:
        path=Path(row['path']).resolve()
        if path in paths or digest(path)!=row['sha256']:raise ValueError('duplicate/changed source binding: '+str(path))
        paths.add(path)
    return paths


def position_conditions(runner):
    """Only declared old condition/config inputs; never inspect result reports."""
    manifest=json.loads(CONDITIONS.read_text());selection=json.loads(SELECTION.read_text())
    if (manifest['schema']!='graph-self-adapter.twin-v02-final-conditions.v1'
        or manifest['condition_seed']!=92005 or manifest['condition_set']!='test_v02'
        or manifest['nominal_special_case'] is not False):raise ValueError('wrong historical position manifest')
    for row in (manifest['condition_generator'],manifest['runtime_contract'],selection['condition_binding']):
        if digest(row['path'])!=row['sha256']:raise ValueError('historical generator/runtime/condition binding changed')
    if Path(selection['condition_binding']['path']).resolve()!=CONDITIONS.resolve():raise ValueError('wrong selected condition manifest')
    plan=json.loads((REPO/'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json').read_text())
    task=runner.task_from_plan(plan,TASK)
    runtime_path=REPO/'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json'
    runtime=runner.validate_runtime_contract(json.loads(runtime_path.read_text()))
    actual={}
    for trial in range(10):
        condition=runner.condition_manifest(task,seed=92005,trial=trial,nominal=False,condition_design=runtime['condition_design'])
        sha=runner.canonical_json_sha256(condition)
        if (manifest['tasks'][TASK].get(str(trial))!=condition
            or manifest['condition_sha256_by_task_trial'][TASK].get(str(trial))!=sha):
            raise ValueError('opened historical cup position differs from original generator: '+str(trial))
        actual[trial]={'condition':condition,'sha256':sha}
    return manifest,selection,actual


def prepare(config,contract,scope):
    runner,packages,solution,paths=core.expected_bindings(TASK,config,contract,scope)
    runner.verify_freeze(solution)
    historical=json.loads(HISTORY.read_text());historical_paths=checked_paths(historical)
    manifest,selection,conditions=position_conditions(runner)
    candidate=json.loads((solution/'CANDIDATE_FREEZE.json').read_text())
    row=runner.validate_test_selection(selection,task=TASK,candidate_id=candidate['candidate_id'],
                                        solution=solution,selection_path=SELECTION)
    for trial,value in conditions.items():
        runner.validate_v02_test_condition(selection,row=row,selection_path=SELECTION,
            runtime_contract_path=REPO/'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json',seed=92005,trial=trial,
            condition=value['condition'])
    paths.update(historical_paths)
    paths.update((Path(__file__),HISTORY,CONDITIONS,SELECTION,Path(manifest['condition_generator']['path']),
                  Path(selection['scope_binding']['path'])))
    return runner,packages,solution,paths,conditions


def validate_campaign(scope_path,freeze_path,*,required_paths,conditions,trial,output,identity):
    scope=json.loads(scope_path.read_text());freeze=json.loads(freeze_path.read_text())
    if (scope.get('execution_authorized') is not True or scope.get('tasks')!=[TASK]
        or scope.get('trials')!=list(range(10)) or scope.get('condition_seed')!=92005
        or scope.get('condition_set')!='test_v02' or scope.get('maximum_full_task_episodes')!=10
        or scope.get('position_only') is not True or scope.get('maximum_hardware_motion_episodes')!=0
        or scope.get('condition_binding')!={'path':str(CONDITIONS.resolve()),'sha256':digest(CONDITIONS)}):
        raise ValueError('scope must authorize exactly ten opened cup position trials')
    if (freeze.get('schema')!=FREEZE_SCHEMA or freeze.get('execution_variant')!=VARIANT
        or freeze.get('status')!='result_free_frozen' or freeze.get('scope_sha256')!=digest(scope_path)):
        raise ValueError('wrong campaign variant, scope binding or result-free freeze')
    bound=checked_paths(freeze)
    if not {p.resolve() for p in required_paths}<=bound:raise ValueError('missing actual historical/new source, condition, graph/model/ref/scene binding')
    jobs=freeze.get('jobs',[])
    if len(jobs)!=10:raise ValueError('freeze requires ten exact jobs')
    seen=set()
    for job in jobs:
        index=job.get('trial')
        if type(index) is not int or index not in range(10) or index in seen:raise ValueError('duplicate or invalid position trial')
        seen.add(index)
        expected={'task':TASK,'trial':index,'seed':92005,'condition_set':'test_v02','fixed_z':True,
                  'condition_sha256':conditions[index]['sha256'],'actual_root_reply_identity':identity}
        if any(job.get(k)!=v or (k=='fixed_z' and type(job.get(k)) is not bool) for k,v in expected.items()):
            raise ValueError('task/condition/Root/fixed-z job mismatch')
        if Path(job.get('output','')).resolve()!=(RUN_ROOT/f'trial_{index:02d}').resolve():
            raise ValueError('job output must be the exact campaign trial directory')
    if output.resolve()!=(RUN_ROOT/f'trial_{trial:02d}').resolve():raise ValueError('actual output differs from frozen trial')
    return {str(p):digest(p) for p in bound}


def validate_config(config_path,contract_path,identity):
    cfg=json.loads(config_path.read_text());contract=json.loads(contract_path.read_text())
    baseline=json.loads((REPO/'evaluations/agent_stage_v06/CONFIG.json').read_text())
    expected={'max_requests_per_episode':16,'max_unknown_observation_retries':2,'hold_sec':.25,
              'wall_timeout_sec':600,'root_identity':identity}
    if (cfg.get('execution_variant')!=VARIANT or contract.get('execution_variant')!=VARIANT
        or cfg.get('default_controller')!='fixed_z' or cfg.get('agent')!=expected or cfg.get('thresholds')!=baseline['thresholds']
        or cfg.get('unknown_holding_timeout_sec')!=baseline['unknown_holding_timeout_sec']
        or contract.get('semantic_fact_ids')!=FACT_IDS):raise ValueError('core sensor or Agent contract differs from frozen v06')
    return RepairConfig(**cfg['thresholds'])


def build_args(runner,packages,solution,*,trial,output):
    candidate=json.loads((solution/'CANDIDATE_FREEZE.json').read_text())
    return runner.parser().parse_args(['--plan',str(REPO/'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json'),
        '--assets',str(REPO/'evaluations/twin_rsi_v02/EXTERNAL_ASSET_BINDINGS.json'),'--packages',str(packages),
        'run','--task',TASK,'--candidate-id',candidate['candidate_id'],'--condition-set','test_v02',
        '--condition-seed','92005','--trial',str(trial),'--selection-freeze',str(SELECTION),
        '--runtime-contract',str(REPO/'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json'),'--output',str(output)])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=(TASK,),default=TASK);p.add_argument('--trial',type=int,choices=range(10),required=True)
    p.add_argument('--condition-set',choices=('test_v02',),default='test_v02')
    p.add_argument('--condition-seed',type=int,choices=(92005,),default=92005)
    p.add_argument('--config',type=Path,default=OWNER/'CONFIG.json');p.add_argument('--contract',type=Path,default=OWNER/'CONTRACT.json')
    p.add_argument('--scope',type=Path,default=OWNER/'SCOPE.json');p.add_argument('--execution-freeze',type=Path)
    p.add_argument('--root-identity',required=True);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);output=a.output or RUN_ROOT/f'trial_{a.trial:02d}'
    if output.exists():raise FileExistsError('fresh campaign output required')
    runner,packages,solution,paths,conditions=prepare(a.config,a.contract,a.scope)
    config=validate_config(a.config,a.contract,a.root_identity)
    ready={'execution_variant':VARIANT,'frozen_core_variant':core.VARIANT,'task':TASK,'trial':a.trial,
        'seed':92005,'condition_set':'test_v02','opened_development_regression':True,'fixed_z':True,'dry_run':not a.execute,
        'output':str(output),'condition':conditions[a.trial]['condition'],'condition_sha256':conditions[a.trial]['sha256'],
        'actual_root_reply_identity':a.root_identity,'required_bindings':[{'path':str(x.resolve()),'sha256':digest(x)} for x in sorted(paths)]}
    if not a.execute:print(json.dumps(ready,indent=2));return ready
    if a.execution_freeze is None:raise ValueError('position result-free execution freeze required; no rollout started')
    bindings=validate_campaign(a.scope,a.execution_freeze,required_paths=paths,conditions=conditions,trial=a.trial,
                               output=output,identity=a.root_identity)
    bindings.update(execution_freeze_sha256=digest(a.execution_freeze),position_campaign_variant=VARIANT,
                    position_condition_sha256=conditions[a.trial]['sha256'])
    worlds=core.install_runtime(runner,task=TASK,packages=packages,config=config,output=output,
                                root_identity=a.root_identity,bindings=bindings)
    old_dump=runner.dump_json
    def campaign_dump(path,report):
        # Core tags its actual implementation first; campaign metadata is additive.
        result=old_dump(path,report)
        if report.get('schema')=='graph-self-adapter.twin-run.v1':
            report.update(execution_variant=VARIANT,frozen_core_variant=core.VARIANT,
                opened_development_regression=True,position_condition_sha256=conditions[a.trial]['sha256'])
            atomic_json(path,report)
        return result
    runner.dump_json=campaign_dump
    try:result=runner.run(build_args(runner,packages,solution,trial=a.trial,output=output))
    except HandoffError as exc:
        world=worlds[-1] if worlds else None;seal=None
        if world:
            seal=world.live_broker.seal(output/'AGENT_PREDICTIONS_SEALED.json',bindings=bindings,counters=world.live_counters())
            world.close()
        result={'execution_variant':VARIANT,'frozen_core_variant':core.VARIANT,'task':TASK,'trial':a.trial,
            'condition_sha256':conditions[a.trial]['sha256'],'strict_success':False,'failure':str(exc),
            'sealed_before_private_evaluator':True,'private_evaluator_not_opened':True,'agent_prediction_seal_sha256':seal}
        atomic_json(output/'AGENT_FAIL_CLOSED.json',result)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
