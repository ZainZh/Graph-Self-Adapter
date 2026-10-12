"""New operator scope for already-qualified expanded tasks; never resets old reservations."""
import json,time,uuid
from pathlib import Path
from prove.xtrainer.agent import write,digest
from prove.xtrainer.io import SafetyError
REPO=Path(__file__).resolve().parents[2]
def new_operator_scope(task,row,registry_path):
    candidate=(REPO/row['candidate_owner']).resolve();root=(REPO/'evaluations/task_expansion_v45').resolve()
    if not candidate.is_relative_to(root) or not (candidate/'ROOT_ACCEPTANCE.json').is_file():raise SafetyError('qualified expansion owner required')
    parent=json.loads((candidate/'TRIAL_SCOPE.json').read_text());registry=json.loads(Path(registry_path).read_text())
    if parent.get('execution_authorized') is not True or parent.get('task')!=task or parent.get('hardware_motion_jobs')!=0:raise SafetyError('expanded task scope authority mismatch')
    if Path(registry['source_acceptance'])!=candidate.relative_to(REPO)/'ROOT_ACCEPTANCE.json':raise SafetyError('registry acceptance binding mismatch')
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8];owner_root=candidate/'operator_runs'/stamp;owner=owner_root/task;owner.mkdir(parents=True);scope={k:v for k,v in parent.items() if k not in ('results','jobs','consumed','ledger')};scope.update(schema='graph-self-adapter.expansion-operator-scope.v1',maximum_full_simulator_episodes=1,hardware_motion_jobs=0,old_experiment_budget_not_reset=True,maximum_paid_API_calls=parent.get('maximum_paid_API_calls',12),user_authorization='explicit fresh simulator invocation of qualified single spoon task; no hardware',operator_source_bindings=[{'executed_path':str(p),'sha256':digest(p)} for p in (Path(__file__).resolve(),Path(__file__).with_name('task_auto_v71.py'),Path(registry_path).resolve())]);path=owner/'TRIAL_SCOPE.json';write(path,scope);return {'scope':str(path),'fresh_owner':str(owner),'owner_root':str(owner_root)}
