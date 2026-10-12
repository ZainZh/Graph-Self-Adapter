"""Each explicit operator sim command gets an independent result-free engineering scope."""
import json,time,uuid
from pathlib import Path
from prove.xtrainer.agent import digest,write
from prove.xtrainer.io import SafetyError

REPO=Path(__file__).resolve().parents[2]

def new_operator_scope(task,row,registry_path,*,repo=REPO):
    candidate=(repo/row['candidate_owner']).resolve()
    if not candidate.is_relative_to((repo/'evaluations/three_task_migration_v25').resolve()):
        raise SafetyError('operator scope must belong to registered task evidence owner')
    parent_scope=candidate/'TRIAL_SCOPE.json'
    parent=json.loads(parent_scope.read_text())
    if parent.get('execution_authorized') is not True or parent.get('task')!=task or parent.get('hardware_motion_jobs')!=0:
        raise SafetyError('registered simulator candidate authority absent')
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    invocation=candidate/'operator_runs'/stamp
    if row['interface']=='ordinary_task_entry':
        owner_root=invocation
        owner=owner_root/task/row['owner_suffix']
    else:
        owner_root=None;owner=invocation
    owner.mkdir(parents=True,exist_ok=False)
    snapshot=owner/'SOURCE_SNAPSHOT'
    bindings=[]
    for source in (Path(__file__),Path(__file__).with_name('three_task_v26.py'),Path(registry_path)):
        saved=snapshot/source.name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(source.read_bytes())
        bindings.append({'executed_path':str(source.resolve()),'snapshot_path':str(saved.resolve()),'sha256':digest(saved)})
    # New operator authority is explicit and distinct from the old experiment's
    # budget. We copy only the established input/safety contract, not its quota,
    # consumed ledger, results or scientific sample design.
    contract_fields=('runtime_inputs','forbidden_runtime_inputs','model','endpoint','judge_fallback',
        'physics_continues_during_API','code_changes_during_episode','seal_before_original_private_goal_and_safety_evaluation')
    scope={key:parent[key] for key in contract_fields if key in parent}
    scope.update(schema='graph-self-adapter.operator-engineering-scope.v1',task=task,mode='sim',execution_authorized=True,
        hardware_motion_jobs=0,maximum_full_simulator_episodes=1,automatic_API_retries=0,user_API_call_limit=None,
        maximum_paid_API_calls=parent.get('API_budget_per_episode',22),API_budget_per_episode=parent.get('API_budget_per_episode',22),
        initial_conditions='Previously opened native nominal scene; independent full native start; engineering rerun, not held-out evidence',
        candidate_budget='One explicitly invoked simulator engineering episode of registered frozen candidate; no source changes during episode',
        parent_candidate_scope=str(parent_scope),parent_candidate_scope_sha256=digest(parent_scope),
        old_experiment_budget_not_reset=True,old_runs_or_scopes_not_resumed=True,
        user_authorization='Explicit manual invocation of authorized three-task simulator workflow; Root authorized automatic independent result-free engineering scopes. No hardware execution.',
        objective='Execute unchanged registered task candidate from native initial state and seal before original final-goal/safety evaluation',
        observer_contract='First grounding and UNKNOWN may require actual current Agent image judgement; this is not unattended execution',
        operator_source_bindings=bindings)
    path=owner/'TRIAL_SCOPE.json';write(path,scope)
    result={'scope':str(path),'scope_sha256':digest(path),'fresh_owner':str(owner),'owner_root':str(owner_root) if owner_root else None,
            'candidate':row['entry_module'],'hardware_connections':0,'episode_started':False,'ledger_reset':False}
    print(json.dumps({'fresh_operator_scope':result}),flush=True)
    return result
