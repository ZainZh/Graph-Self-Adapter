from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/deployment")]

from multitask_dual_tool_server_fair_v22_1 import main, optional_table_contract, validate_server_environment


@pytest.mark.parametrize("task,essential", [
    ("sorting_cube_bimanual", {"red_cube", "yellow_cube", "tray"}),
    ("sweep_objects", {"broom", "dustpan", "yellow_cube"}),
])
def test_server_check_uses_task_roles_and_fair_candidate_without_io(task, essential):
    report = main(["--task", task, "check"])
    assert set(report["roles"]["SAM_roles"]) == essential
    assert report["roles"]["optional_SAM_roles"] == ["table_patch"]
    assert report["server_candidate"] == "fair_io_v22_1"
    assert report["device_connections"] == report["api_calls"] == 0
    assert report["episode_started"] is False


def test_environment_requires_exact_gpu3_and_thread_limits(monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert validate_server_environment()["CUDA_VISIBLE_DEVICES"] == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(SystemExit, match="GPU3|CUDA_VISIBLE_DEVICES=3"):
        validate_server_environment()


def test_optional_table_never_removes_actor_or_target_roles():
    contract = {"SAM_roles":["broom","dustpan","yellow_cube","table_patch"], "grounding":"fresh"}
    result = optional_table_contract(contract, "sweep_objects")
    assert result["SAM_roles"] == ["broom","dustpan","yellow_cube"]
    assert result["optional_SAM_roles"] == ["table_patch"]
