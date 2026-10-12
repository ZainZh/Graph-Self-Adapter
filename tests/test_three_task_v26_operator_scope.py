from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from three_task_v26_operator_scope import new_operator_scope

def test_fresh_operator_scopes_do_not_reset_consumed_experiment(tmp_path):
    candidate=tmp_path/'evaluations/three_task_migration_v25/put_mug_on_box/candidate';candidate.mkdir(parents=True)
    old={'execution_authorized':True,'task':'put_mug_on_box','hardware_motion_jobs':0,'maximum_full_simulator_episodes':1,'API_budget_per_episode':22}
    scope=candidate/'TRIAL_SCOPE.json';scope.write_text(json.dumps(old));ledger=candidate/'FORMAL_RUN_LEDGER.json';ledger.write_text('{"runs":[{"consumed":true}]}')
    registry=tmp_path/'REGISTRY.json';registry.write_text('{}')
    row={'candidate_owner':'evaluations/three_task_migration_v25/put_mug_on_box/candidate','entry_module':'candidate','interface':'cup_one_command'}
    first=new_operator_scope('put_mug_on_box',row,registry,repo=tmp_path)
    second=new_operator_scope('put_mug_on_box',row,registry,repo=tmp_path)
    assert first['scope']!=second['scope']
    assert json.loads(scope.read_text())==old and json.loads(ledger.read_text())=={'runs':[{'consumed':True}]}
    new=json.loads(Path(first['scope']).read_text())
    assert new['maximum_full_simulator_episodes']==1 and new['old_experiment_budget_not_reset'] is True
    assert len(new['operator_source_bindings'])==3
