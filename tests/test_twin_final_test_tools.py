import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = _module("_twin_run_final_test_under_test",
              "tools/task_graph/twin_run_final_test.py")
SUMMARY = _module("_twin_summarize_final_test_under_test",
                  "tools/task_graph/twin_summarize_final_test.py")
REVIEW = _module("_twin_adjudicate_execution_under_test",
                 "tools/task_graph/twin_adjudicate_execution.py")


def test_real_frozen_plan_has_mixed_snapshots_and_does_not_create_outputs(tmp_path):
    selected_output = tmp_path / "selected"
    baseline_output = tmp_path / "initial_graph"
    plan = RUN.build_plan(
        ROOT / "evaluations/twin_rsi_v01/FINAL_SELECTION.json",
        ROOT / "evaluations/twin_rsi_v01/INITIAL_GRAPH_SELECTION.json",
        ROOT / "evaluations/twin_rsi_v01/FINAL_TEST_SCOPE.json",
        selected_output, baseline_output)

    assert plan["job_counts"] == {"selected": 200, "baseline": 100, "total": 300}
    assert {row["candidate_id"] for row in plan["selected"]["tasks"]} == {
        "r2_draft01", "r3_draft01"}
    assert {row["candidate_id"] for row in plan["baseline"]["tasks"]} == {
        "r0_common_backend"}
    spoon = next(job for job in plan["jobs"] if job["cohort"] == "selected" and
                 job["task"] == "sort_two_spoon")
    assert "r2_draft01/source_snapshot/tools/task_graph/twin_rsi.py" in spoon["command"][1]
    assert "--selection-freeze" in spoon["command"]
    assert not selected_output.exists() and not baseline_output.exists()


def _run_report(success: bool, *, condition_hash: str = "same"):
    return {
        "strict_success": success,
        "condition": {"task": "task", "trial": 0},
        "selection_row": {"task": "task"},
        "bindings": {
            "condition_sha256": condition_hash, "graph_sha256": "g",
            "model_sha256": "m", "candidate_freeze_sha256": "f",
            "selection_freeze_sha256": "s",
        },
        "trace": [{"controller": {
            "latent": {"loss_before": 0.2, "loss_after": 0.1,
                       "gradient_local": [0.1, 0.0, 0.0], "admitted": True},
            "action_online_minus_fixed_l2_rad": 0.03,
        }}],
    }


def test_selected_pairing_accepts_per_task_candidate_and_requires_matched_input():
    selection = {"candidate_id": "per_task_selected_R0_to_R3",
                 "tasks": [{"task": "task", "candidate_id": "r2_draft01"}]}
    fixed, online = _run_report(False), _run_report(True)
    reports = {("task", "fixed_z", 0): (Path("fixed.json"), fixed),
               ("task", "online_z", 0): (Path("online.json"), online)}
    paired = SUMMARY.selected_paired(selection, reports, 1)
    assert paired["task_candidates"] == {"task": "r2_draft01"}
    assert paired["outcomes"] == {"online_only": 1}
    assert paired["tasks"][0]["online_gradient_updates"] == 1
    assert paired["tasks"][0]["mean_action_change_rad"] == pytest.approx(0.03)

    online["bindings"]["model_sha256"] = "different"
    with pytest.raises(ValueError, match="mismatches model_sha256"):
        SUMMARY.selected_paired(selection, reports, 1)


def test_pairing_does_not_count_loss_without_an_admitted_gradient():
    selection = {"candidate_id": "selected",
                 "tasks": [{"task": "task", "candidate_id": "candidate"}]}
    fixed, online = _run_report(False), _run_report(False)
    online["trace"][0]["controller"]["latent"] = {
        "loss_before": 0.2, "loss_after": 0.1,
        "gradient_local": None, "admitted": False}
    reports = {("task", "fixed_z", 0): (Path("fixed.json"), fixed),
               ("task", "online_z", 0): (Path("online.json"), online)}
    paired = SUMMARY.selected_paired(selection, reports, 1)
    assert paired["tasks"][0]["online_gradient_updates"] == 0
    assert paired["tasks"][0]["updated_loss_nonincrease_fraction"] is None


def test_strict_success_recomputed_from_all_three_gates():
    report = {"graph_completed": True, "graph_failure": None,
              "original_task_outcome": {
                  "numerical_safety": {"passed": True},
                  "task_outcome": {"passed": True}}}
    assert SUMMARY._strict_success(report)
    report["original_task_outcome"]["task_outcome"]["passed"] = False
    assert not SUMMARY._strict_success(report)


def test_execution_review_accepts_only_one_exact_qacc_warning():
    text = ('{"failure":"stage_budget_without_boolean_completion",'
            '"output":"/tmp/report.json","strict_success":false}\n'
            'WARNING: Nan, Inf or huge value in QACC at DOF 26. '
            'The simulation is unstable. Time = 0.2290.\n\n')
    payload, warning = REVIEW.parse_stdout(text)
    assert payload["strict_success"] is False
    assert warning == {
        "text": ("WARNING: Nan, Inf or huge value in QACC at DOF 26. "
                 "The simulation is unstable. Time = 0.2290."),
        "dof": 26, "physical_time_sec": pytest.approx(0.229),
        "diagnostic": "numerical_instability"}
    with pytest.raises(ValueError, match="single approved"):
        REVIEW.parse_stdout(text + "another warning\n")
    with pytest.raises(ValueError, match="single approved"):
        REVIEW.parse_stdout(text.replace("QACC", "QPOS"))


def test_execution_review_cannot_admit_a_strict_success(tmp_path):
    report_path = tmp_path / "report.json"
    report = {
        "schema": SUMMARY.RUN_SCHEMA, "task": "task", "candidate_id": "candidate",
        "condition_set": "test", "controller": "fixed_z",
        "condition": {"task": "task", "trial": 0, "seed": 71005},
        "strict_success": True, "graph_completed": True, "graph_failure": None,
        "original_task_outcome": {"numerical_safety": {"passed": True},
                                  "task_outcome": {"passed": True}}}
    payload = {"output": str(report_path), "strict_success": True, "failure": None}
    row = {"task": "task", "candidate_id": "candidate", "controller": "fixed_z",
           "trial": 0}
    with pytest.raises(ValueError, match="restricted to strict failures"):
        REVIEW._validate_report(report_path, report, payload, row,
                                tmp_path / "selection.json")
