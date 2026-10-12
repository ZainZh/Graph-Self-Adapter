import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/deployment"))

import multitask_server_fair_entry_v22_3 as entry


def test_loader_selects_fair_backend_only_for_sim_io():
    sentinel = object()
    loader = entry.fair_backend_loader(lambda path, name: sentinel)
    module = loader(ROOT / "tools/deployment/sim_robot_io.py", "sim")
    assert module.SimRobot.__name__ == "FairSimRobot"
    assert loader(ROOT / "tools/deployment/other.py", "other") is sentinel


def test_server_environment_requires_explicit_bounded_threads(monkeypatch):
    for key in entry.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "2")
    report = entry.validate_server_environment()
    assert report["CUDA_VISIBLE_DEVICES"] == "2"
    assert report["MUJOCO_EGL_DEVICE_ID"] == "2"
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    with pytest.raises(SystemExit, match="thread limits"):
        entry.validate_server_environment()


@pytest.mark.parametrize("task,required", [
    ("open_drawer_and_place_object", ["drawer", "drawer_shell", "red_cube"]),
    ("hanging_mug", ["mug", "rack"]),
])
def test_server_check_is_no_io_and_task_specific(task, required):
    report = entry.main(["--task", task, "check"])
    assert report["device_connections"] == report["api_calls"] == 0
    assert report["roles"]["SAM_roles"] == required
    assert report["roles"]["optional_SAM_roles"] == ["table_patch"]
