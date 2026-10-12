import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/evidence/build_twin_final_videos.py"
SPEC = importlib.util.spec_from_file_location("_twin_final_videos_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VIDEOS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VIDEOS
SPEC.loader.exec_module(VIDEOS)


TASKS = [
    "collect_apple", "collect_cubes", "hanging_mug",
    "open_drawer_and_place_object", "put_knife_board_new", "put_mug_on_box",
    "sort_two_spoon", "sorting_cube_bimanual", "stack_cups", "sweep_objects",
]


def _report(task, trial, success, controller):
    condition = {"task": task, "trial": trial, "seed": 71005}
    return {"strict_success": success, "condition": condition,
            "bindings": {"condition_sha256": f"condition-{task}-{trial}",
                         "graph_sha256": f"graph-{task}",
                         "model_sha256": f"model-{task}",
                         "candidate_freeze_sha256": f"freeze-{task}"},
            "controller": controller}


def test_case_selection_is_bounded_first_and_semantically_paired():
    selected, baseline = {}, {}
    for task in TASKS:
        for trial in range(3):
            fixed = _report(task, trial, trial == 1, "fixed_z")
            online = _report(task, trial, trial == 1, "online_z")
            selected[(task, "fixed_z", trial)] = (Path(f"{task}_{trial}_fixed.json"), fixed)
            selected[(task, "online_z", trial)] = (Path(f"{task}_{trial}_online.json"), online)
            baseline[(task, "fixed_z", trial)] = (
                Path(f"baseline_{task}_{trial}.json"),
                _report(task, trial, False, "fixed_z"))

    cases = VIDEOS.choose_selected(selected, TASKS, 3)
    assert len(cases) == 13
    assert all(case["report_data"]["condition"]["trial"] == 1
               for case in cases[:10])
    assert {case["task"] for case in cases[10:]} == {
        "put_knife_board_new", "put_mug_on_box", "sorting_cube_bimanual"}
    assert all(case["report_data"]["strict_success"] is False for case in cases[10:])

    outer = VIDEOS.choose_comparisons(baseline, selected, 3)
    assert [(case["task"], case["trial"]) for case in outer] == [
        ("collect_cubes", 1), ("open_drawer_and_place_object", 1),
        ("sweep_objects", 1)]

    # Make the first bimanual discordance fixed failure / online success.
    selected[("sorting_cube_bimanual", "fixed_z", 1)][1]["strict_success"] = False
    inner = VIDEOS.choose_inner_comparison(selected, 3)
    assert inner["trial"] == 1
    assert inner["comparison_kind"] == "inner_fixed_vs_online"
    assert "69% vs 70%" in inner["caption"]
