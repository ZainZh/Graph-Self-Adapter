import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('_multitask_registry',ROOT/'tools/task_graph/multitask_entry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_ten_distinct_task_assets_do_not_assume_cup_graph_or_model():
    _,rows=m.registry();assert len(rows)==10 and 'fold_cloth' not in rows and 'hanging_mug_bimanual' not in rows
    for task,row in rows.items():
        result=m.check_task(task);assert result['task_assets_verified'] and result['historical_graph_nodes']==row['historical_stage_count']
    assert rows['collect_cubes']['historical_stage_count']==35 and 'Pull' in rows['open_drawer_and_place_object']['phases']

def test_missing_executor_refuses_instead_of_silent_cup_fallback():
    with pytest.raises(ValueError,match='task-specific executor/judge missing'):m.command('collect_apple','sim',None,None,None,None)
    with pytest.raises(ValueError,match='unknown/unqualified'):m.check_task('fold_cloth')

def test_existing_cup_adapter_does_not_launch_old_scope_without_new_freeze():
    with pytest.raises(ValueError,match='new explicit'):m.command('put_mug_on_box','sim',None,None,None,None)
