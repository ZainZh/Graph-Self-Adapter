#!/usr/bin/env python3
"""v06 continuous native-initial-state Agent semantics; dry-run by default."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from tools.task_graph import twin_sensor_rsi as frozen_sensor
from tools.task_graph.twin_sensor_rsi_repair1 import REPAIR_SOURCES
from tools.task_graph.twin_sensor_transition_rsi import SOURCES as TRANSITION_SOURCES
from prove.task_graph.twin_sensor_judge_repair1 import RepairConfig
from prove.task_graph.twin_sensor_transition_adapter import routed_holding_contract
from prove.task_graph.agent_live_adapter import install_agent_world, semantic_contract, PERMISSION
from prove.task_graph.agent_live_loop import HandoffError, atomic_json, VARIANT
from prove.task_graph.agent_stage_judge import digest

LIVE_SOURCES = ('prove/task_graph/agent_live_loop.py', 'prove/task_graph/agent_live_adapter.py',
                'tools/task_graph/twin_agent_live_rsi.py', 'prove/task_graph/agent_stage_judge.py')
BASE_SOURCES = ('tools/task_graph/twin_sensor_rsi.py', 'prove/task_graph/twin_sensor_adapter.py',
                'prove/task_graph/twin_sensor_judge.py', *REPAIR_SOURCES, *TRANSITION_SOURCES)


def expected_bindings(task, config, contract, scope):
    runner, snapshot, packages = frozen_sensor.load_frozen_runner(task)
    solution = packages / ('twin_' + task) / 'solution'
    plan_path = REPO / 'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json'
    assets_path = REPO / 'evaluations/twin_rsi_v02/EXTERNAL_ASSET_BINDINGS.json'
    plan = json.loads(plan_path.read_text())
    task_row = runner.task_from_plan(plan, task)
    paths = {REPO / p for p in (*LIVE_SOURCES, *BASE_SOURCES)}
    paths.update(snapshot.rglob('*.py'))
    paths.update(p for p in solution.rglob('*') if p.is_file())
    paths.update((config, contract, scope, plan_path, assets_path,
                  REPO / 'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json'))
    # Complete source asset catalog includes scene, calibration, motion and scene dependencies.
    for row in task_row['sources'].values():
        paths.add(Path(row['path']))
    catalog = json.loads(assets_path.read_text())
    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and isinstance(value['path'], str):
                paths.add(Path(value['path']))
            for key, item in value.items():
                if isinstance(key, str) and key.startswith('/') and isinstance(item,str) and len(item)==64:
                    paths.add(Path(key))
                walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(catalog)
    return runner, packages, solution, paths


def validate_live_execution(scope_path, freeze_path, *, task, output, root_identity, required_paths):
    scope = json.loads(scope_path.read_text()); freeze = json.loads(freeze_path.read_text())
    if scope.get('execution_authorized') is not True or freeze.get('schema') != 'graph-self-adapter.agent-live-freeze.v1':
        raise ValueError('explicit v06 authorization and result-free freeze required')
    if freeze.get('execution_variant') != VARIANT or freeze.get('status') != 'result_free_frozen':
        raise ValueError('wrong variant or non-result-free freeze')
    if freeze.get('scope_sha256') != digest(scope_path):
        raise ValueError('scope hash mismatch')
    if (task not in scope.get('tasks', []) or scope.get('condition_seed') != 51005
        or scope.get('trials') != [0] or scope.get('maximum_full_task_episodes') != 2
        or len(freeze.get('jobs', [])) > 2
        or len({job.get('task') for job in freeze.get('jobs',[])}) != len(freeze.get('jobs',[]))):
        raise ValueError('freeze jobs exceed the two native development tasks authorized by scope')
    bound = {}
    for binding in freeze['bindings']:
        path = Path(binding['path']).resolve()
        if path in bound or digest(path) != binding['sha256']:
            raise ValueError('duplicate or mismatching binding: ' + str(path))
        bound[path] = binding['sha256']
    if not {p.resolve() for p in required_paths} <= set(bound):
        missing = {str(p.resolve()) for p in required_paths if p.resolve() not in bound}
        raise ValueError('incomplete actual source/graph/model/reference/scene/calibration binding: ' + str(sorted(missing)))
    jobs = [job for job in freeze['jobs'] if job.get('task') == task and job.get('trial') == 0
            and job.get('seed') == 51005 and job.get('fixed_z') is True
            and job.get('native_initial_state') is True and Path(job['output']).resolve() == output.resolve()
            and job.get('actual_root_reply_identity') == root_identity]
    if len(jobs) != 1:
        raise ValueError('unique exact native-initial-state fixed-z task/output/Root job absent')
    return {str(p): sha for p, sha in bound.items()}


def install_runtime(runner, *, task, packages, config, output, root_identity, bindings):
    worlds = []; physics = runner._physics()
    solution = packages / ('twin_' + task) / 'solution'
    references = json.loads((solution / 'refs/visible_centroids.json').read_text())
    Base = install_agent_world(physics, config=config, references=references, judge_mode='sensor_only',
        seal_path=output / 'SENSOR_PREDICTIONS_SEALED.json', timeout_sec=1., provenance=bindings,
        broker_options={'root': output / 'AGENT_HANDOFFS', 'task': task,
                        'graph_sha256': digest(solution / 'graph/task_graph.json'), 'root_identity': root_identity,
                        'wall_timeout_sec':600., 'max_requests':16}, bindings=bindings)
    class RegisteredWorld(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); worlds.append(self)
    proxy = SimpleNamespace(**{k:v for k,v in vars(physics).items() if not k.startswith('__')})
    proxy.TwinWorld = RegisteredWorld; runner._physics = lambda: proxy
    runner.holding_fact = lambda contract, raw, **kw: routed_holding_contract(
        {**(worlds[-1].sensor_stage or {}), 'holding_contract':contract,
         'target':kw.get('target_role'), 'goal_relation':kw.get('goal_relation')}, raw)
    original_clock = runner.StageReferenceClock
    class HoldExcludedClock(original_clock):
        def at(self, physical_elapsed):
            world = worlds[-1] if worlds else None
            sid = (world.sensor_stage or {}).get('stage_id') if world else None
            held = world.live_hold_by_stage.get(sid, 0.) if world else 0.
            return super().at(max(0., physical_elapsed - held))
    runner.StageReferenceClock = HoldExcludedClock
    original_runtime = runner.GRAPH.TwinGraphRuntime
    class LiveRuntime(original_runtime):
        @property
        def stage(self):
            stage = super().stage
            if stage['phase'] in ('Align','Release'):
                stage = {**stage, 'done': [*stage['done'], 'semantic.goal']}
            if worlds:
                world = worlds[-1]
                if world.sensor_stage is not None and world.sensor_stage['stage_id'] != stage['stage_id']:
                    world.sensor_unknown_since = None
                world.sensor_stage = stage; world.sensor_runtime = self
            return stage
        def transition(self, facts):
            if worlds and worlds[-1].sensor_motion_blocked: return False
            return super().transition(facts)
    runner.GRAPH.TwinGraphRuntime = LiveRuntime
    original_query = runner.query_stage_goal_predicates
    def query(world, stage, rows):
        if stage['phase'] == 'Release': return {row['id']: None for row in rows}
        return original_query(world, stage, rows)
    runner.query_stage_goal_predicates = query
    original_contract = runner.evaluate_stage_contract
    def evaluate(stage, **kwargs):
        original_facts, diagnostics = original_contract(stage, **kwargs)
        if stage['phase'] not in ('Align','Release'): return original_facts, diagnostics
        world = worlds[-1]; before = world.proprioception(); before_time = float(world.data.time)
        facts, diagnostics = semantic_contract(stage, original_facts, diagnostics, ask=world.live_ask)
        if float(world.data.time) != before_time:
            # UNKNOWN pose holds do not authorize stale release or actual joint gates.
            frame = world.observe_visible(); prop = world.proprioception()
            refreshed = dict(kwargs)
            refreshed['raw'] = world.visible_stage_facts(stage['object'], frame, arm=stage['arm'],
                support=stage['target'], goal_relation=stage['goal_relation'])
            index = 0 if stage['arm']=='left' else 1
            drift = float(np.max(np.abs(prop.joint_pos[index]-before.joint_pos[index])))
            if kwargs.get('actor_joint_error_max_rad') is not None:
                refreshed['actor_joint_error_max_rad'] = kwargs['actor_joint_error_max_rad'] + drift
            refacts, rediagnostics = original_contract(stage, **refreshed)
            for row in stage.get('completion_predicates', []) + stage.get('invariant_predicates', []):
                if row['predicate'] in ('actor_command_reached','released','finite_simulation'):
                    facts[row['id']] = refacts[row['id']]
            diagnostics.extend({**d, 'after_unknown_hold':True} for d in rediagnostics)
        if stage['phase']=='Align' and stage['holding_contract']=='held_or_supported':
            semantic = next((d for d in reversed(diagnostics) if d.get('id')=='semantic.goal'), {})
            permission = semantic.get('facts',{}).get(PERMISSION)
            if permission is not None:
                world.live_permission = (stage['stage_id'], float(world.data.time), permission)
                for row in stage.get('completion_predicates',[]) + stage.get('invariant_predicates',[]):
                    if row['predicate']=='holding': facts[row['id']] = permission
                if permission is not True: facts['semantic.goal'] = False
        return facts, diagnostics
    runner.evaluate_stage_contract = evaluate
    dump = runner.dump_json
    def write(path, report):
        if report.get('schema') == 'graph-self-adapter.twin-run.v1':
            world = worlds[-1]
            report.update(execution_variant=VARIANT, agent_prediction_seal_sha256=world.live_seal_hash,
                actual_root_reply_identity=root_identity, live_integrity_counters=world.live_counters(),
                graph_frozen_sha256=digest(solution/'graph/task_graph.json'), actual_runtime_bindings=bindings)
        return dump(path, report)
    runner.dump_json = write
    return worlds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('put_mug_on_box','put_knife_board_new'), required=True)
    p.add_argument('--scope',type=Path,required=True); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True); p.add_argument('--execution-freeze',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--root-identity',required=True)
    p.add_argument('--execute',action='store_true'); a = p.parse_args(argv)
    if a.output.exists(): raise FileExistsError('fresh output required')
    runner, packages, solution, paths = expected_bindings(a.task,a.config,a.contract,a.scope)
    runner.verify_freeze(solution)
    ready = {'execution_variant':VARIANT,'task':a.task,'trial':0,'seed':51005,'fixed_z':True,
             'native_initial_state':True,'dry_run':not a.execute,'output':str(a.output),
             'required_bindings':[{'path':str(x.resolve()),'sha256':digest(x)} for x in sorted(paths)],
             'actual_root_reply_identity':a.root_identity}
    if not a.execute: print(json.dumps(ready,indent=2)); return ready
    if a.execution_freeze is None: raise ValueError('v06 result-free freeze required; no rollout started')
    bindings = validate_live_execution(a.scope,a.execution_freeze,task=a.task,output=a.output,
                                      root_identity=a.root_identity,required_paths=paths)
    cfg = json.loads(a.config.read_text()); contract = json.loads(a.contract.read_text())
    if (cfg.get('execution_variant') != VARIANT or contract.get('execution_variant') != VARIANT
        or contract.get('semantic_fact_ids') != ['semantic.continue_placing_permission','semantic.align_ready','semantic.placement_complete']):
        raise ValueError('explicit v06 configuration and semantic fact mapping required')
    if cfg.get('agent') != {'max_requests_per_episode':16,'max_unknown_observation_retries':2,
                           'hold_sec':.25,'wall_timeout_sec':600,'root_identity':a.root_identity}:
        raise ValueError('actual Root identity or bounded handoff parameters differ from frozen config')
    config = RepairConfig(**cfg['thresholds'])
    bindings['execution_freeze_sha256'] = digest(a.execution_freeze)
    worlds = install_runtime(runner,task=a.task,packages=packages,config=config,output=a.output,
                             root_identity=a.root_identity,bindings=bindings)
    freeze = json.loads((solution/'CANDIDATE_FREEZE.json').read_text())
    args = runner.parser().parse_args(['--plan',str(REPO/'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json'),
        '--assets',str(REPO/'evaluations/twin_rsi_v02/EXTERNAL_ASSET_BINDINGS.json'),'--packages',str(packages),
        'run','--task',a.task,'--candidate-id',freeze['candidate_id'],'--condition-set','development_v02',
        '--condition-seed','51005','--trial','0','--runtime-contract',str(REPO/'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json'),
        '--output',str(a.output)])
    try: result = runner.run(args)
    except HandoffError as exc:
        world = worlds[-1] if worlds else None
        seal_hash = None
        if world:
            seal_hash = world.live_broker.seal(a.output/'AGENT_PREDICTIONS_SEALED.json',bindings=bindings,counters=world.live_counters())
            world.close()
        result = {'execution_variant':VARIANT,'task':a.task,'strict_success':False,'failure':str(exc),
                  'sealed_before_private_evaluator':True,'private_evaluator_not_opened':True,
                  'agent_prediction_seal_sha256':seal_hash}
        atomic_json(a.output/'AGENT_FAIL_CLOSED.json',result)
    print(json.dumps(result,indent=2)); return result

if __name__=='__main__': main()
