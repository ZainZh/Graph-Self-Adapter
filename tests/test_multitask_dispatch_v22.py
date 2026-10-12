import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('_dispatch_v22',ROOT/'tools/task_graph/multitask_entry_v22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_all_ten_are_explicit_independent_leaf_routes():
    rows=m.listing();assert len(rows)==10 and all(row['leaf_present'] for row in rows)
    for task in m.TASKS:
        command=m.leaf_command(task,'check');assert task in command[1] and command[-1]=='check'
    with pytest.raises(ValueError,match='fallback'):m.leaf_command('fold_cloth','real')


def test_list_cannot_claim_qualification_from_historical_assets_or_file_presence():
    for row in m.listing():assert row['launch_file_presence_is_not_algorithm_success'] and 'qualification_from_parallel_STATUS' in row


def test_ordinary_and_cup_frontdoors_dispatch_real_to_bound_adapter_not_sim_or_echo():
    for task in ['collect_apple','collect_cubes','put_knife_board_new','sort_two_spoon']:
        text=(ROOT/'task_packages/multitask_parallel_v22'/task/'run.sh').read_text()
        assert 'ordinary_real_entry_v21.py' in text and 'MODE=execute' in text and task in text
    cup=(ROOT/'task_packages/multitask_parallel_v22/put_mug_on_box/run.sh').read_text()
    assert 'promoted_gemini_real_entry.py' in cup and 'trajectory_one_command_v19.py' in cup
