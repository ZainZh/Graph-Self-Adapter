import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/deployment"))

from multitask_hanging_repair_v22_2 import optional_table_role_contract


def test_table_patch_optional_but_hanging_actor_target_stay_required():
    repaired = optional_table_role_contract({"SAM_roles": ["mug", "rack", "table_patch"], "fixed_declared_anchor_roles": []})
    assert repaired["SAM_roles"] == ["mug", "rack"]
    assert repaired["optional_SAM_roles"] == ["table_patch"]


def test_repair_entry_check_is_no_io_and_reports_reduced_required_roles():
    import multitask_hanging_repair_entry_v22_2 as entry
    report = entry.main(["check"])
    assert report["device_connections"] == report["api_calls"] == 0
    assert report["roles"]["SAM_roles"] == ["mug", "rack"]
    assert report["roles"]["optional_SAM_roles"] == ["table_patch"]
