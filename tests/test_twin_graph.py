import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graph = load("test_twin_graph_module", "prove/task_graph/twin_graph.py")
skill = load("test_twin_skill_module", "prove/fast_adapter/twin_local_skill.py")
runner = load("test_twin_rsi_module", "tools/task_graph/twin_rsi.py")


def test_root_plan_compiles_all_cycles_and_preserves_end_hold():
    compiled = graph.compile_root_plan(ROOT / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json")
    assert set(compiled) == {"collect_apple", "collect_cubes", "hanging_mug",
                             "open_drawer_and_place_object", "put_knife_board_new",
                             "put_mug_on_box", "sort_two_spoon", "sorting_cube_bimanual",
                             "stack_cups", "sweep_objects"}
    assert sum(len(x["stages"]) for x in compiled.values()) == 166
    assert [x["phase"] for x in compiled["sweep_objects"]["stages"]][-1] == "Hold"
    apple = compiled["collect_apple"]["stages"]
    assert apple[2]["reference"]["end_time_sec"] < 3.97
    release = next(x for x in apple if x["phase"] == "Release")
    assert release["done"] == ["released", "object_stabilized", "inside", "reference_progress_reached"]
    assert apple[-1]["phase"] == "Wait"
    assert all(s["reference"]["progress_only"] for s in apple)


def test_surface_centroid_uses_depth_mask_and_calibration():
    depth = np.ones((3, 3), dtype=float)
    mask = np.zeros((3, 3), dtype=bool); mask[1:, 1:] = True
    K = np.eye(3); K[0, 2] = K[1, 2] = 1
    value, record = graph.surface_centroid(depth, mask, K, np.eye(4), min_points=4)
    assert record["known"]
    np.testing.assert_allclose(value, [0.5, 0.5, 1.0])


def test_local_z_missing_frame_does_not_repeat_update_and_changes_action(tmp_path):
    input_dim, output_dim = 6, 2
    metadata = {"schema": skill.FrozenNumpyMLP.schema, "z_feature_slice": [3, 6], "action_limit_rad": 1.0}
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "input_mean": np.zeros(input_dim), "input_scale": np.ones(input_dim),
        "output_mean": np.zeros(output_dim), "output_scale": np.ones(output_dim),
        "w0": np.zeros((64, input_dim)), "b0": np.zeros(64),
        "w1": np.zeros((128, 64)), "b1": np.zeros(128),
        "w2": np.zeros((output_dim, 128)), "b2": np.zeros(output_dim),
    }
    arrays["w0"][0, 3] = 1.0; arrays["w1"][0, 0] = 1.0; arrays["w2"][0, 0] = 1.0
    path = tmp_path / "skill.npz"; np.savez(path, **arrays)
    runtime = skill.TwinLocalSkill(path, online_z=True)
    first = runtime.command(np.zeros(input_dim), current_surface_xyz=[0.01, 0, 0], reference_surface_xyz=[0, 0, 0])
    updates = runtime.latent.updates
    missing = runtime.command(np.zeros(input_dim), current_surface_xyz=None, reference_surface_xyz=None)
    assert runtime.latent.updates == updates
    assert first.record["action_online_minus_fixed_l2_rad"] > 0
    assert missing.record["latent"]["measurement_known"] is False
    assert first.record["latent"]["loss_after"] <= first.record["latent"]["loss_before"]


def test_r1_override_compiles_exact_root_contracts():
    compiled = graph.compile_root_plan(
        ROOT / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json",
        ROOT / "evaluations/twin_rsi_v01/R1_OVERRIDES.json")
    assert sum(len(x["stages"]) for x in compiled.values()) == 161
    drawer = compiled["open_drawer_and_place_object"]["stages"]
    assert drawer[2]["phase"] == "Pull"
    assert drawer[2]["holding_contract"] == "loaded_contact"
    assert any(x["predicate"] == "visible_displacement" and x["absolute"] for x in drawer[2]["completion_predicates"])
    knife_align = next(x for x in compiled["put_knife_board_new"]["stages"] if x["phase"] == "Align")
    assert knife_align["holding_contract"] == "held_or_supported"
    assert "Lift" not in [x["phase"] for x in compiled["sweep_objects"]["stages"]]


def test_visible_cache_is_explicit_and_expires():
    cache = runner.VisibleCentroidCache(.5)
    frame = {"physical_time_sec": 1.0, "roles": {"apple": {"centroid_world_m": [1, 2, 3]}}}
    value, state = cache.get("apple", frame)
    np.testing.assert_allclose(value, [1, 2, 3]); assert state["fresh"]
    held = {"physical_time_sec": 1.4, "roles": {"apple": {"centroid_world_m": None}}}
    value, state = cache.get("apple", held)
    np.testing.assert_allclose(value, [1, 2, 3]); assert state["held_previous"] and not state["fresh"]
    expired = {"physical_time_sec": 1.51, "roles": {"apple": {"centroid_world_m": None}}}
    value, state = cache.get("apple", expired)
    assert value is None and not state["known"] and state["reason"] == "last_valid_expired"


def test_public_stability_requires_fresh_stationary_measurements():
    tracker = runner.PublicVisualStability(.005)
    def frame(t, value):
        return {"physical_time_sec": t, "roles": {"drawer": {"centroid_world_m": value}}}
    assert not tracker.update("drawer", frame(1.0, [0, 0, 0]))
    assert tracker.update("drawer", frame(1.21, [.001, 0, 0]))
    assert not tracker.update("drawer", frame(1.3, [.02, 0, 0]))
    assert not tracker.update("drawer", frame(1.6, None))


def test_r1_predicates_preserve_unknown_and_relative_geometry():
    stage = {"object": "drawer", "target": "drawer_shell", "holding_contract": "loaded_contact",
             "goal_relation": "open"}
    relation = {"object_tcp_end": [0.1, 0, 0], "object_target_end": [0.2, 0, 0]}
    value, _ = runner.evaluate_contract_predicate(
        {"predicate": "object_target_relation", "tolerance_m": .02}, stage=stage, progress=1,
        raw={}, object_value=None, target_value=np.zeros(3), tcp=np.zeros(3),
        relation_reference=relation, entry_geometry={})
    assert value is None
    value, diagnostic = runner.evaluate_contract_predicate(
        {"predicate": "visible_displacement", "role": "drawer", "support": "drawer_shell",
         "axis": "x", "min_m": .035, "absolute": True}, stage=stage, progress=1, raw={},
        object_value=np.array([-.04, 0, 0]), target_value=np.zeros(3), tcp=np.zeros(3),
        relation_reference=relation,
        entry_geometry={"drawer": np.zeros(3), "drawer_shell": np.zeros(3)})
    assert value is True and diagnostic["displacement_m"] == .04


def test_handover_anchors_come_from_distinct_release_endpoints():
    compiled = graph.compile_root_plan(
        ROOT / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json",
        ROOT / "evaluations/twin_rsi_v01/R1_OVERRIDES.json")
    path = ROOT / "runs/twin_rsi_v01/r0_draft01/packages/twin_put_mug_on_box/solution/refs/visible_centroids.json"
    revised = runner.augment_reference_contracts(compiled["put_mug_on_box"], json.loads(path.read_text()))
    anchors = revised["handover_anchors"]
    assert anchors["c01"]["source_stage_id"].endswith("release")
    assert anchors["c03"]["source_stage_id"].endswith("release")
    assert anchors["c01"]["centroid_world_m"] != anchors["c03"]["centroid_world_m"]


def test_reference_refresh_defaults_to_observer_hash_change():
    decide = runner.observer_reference_refresh
    assert decide(parent_observer_sha256="old", current_observer_sha256="new", requested=None)
    assert decide(parent_observer_sha256=None, current_observer_sha256="new", requested=None)
    assert not decide(parent_observer_sha256="same", current_observer_sha256="same", requested=None)
    assert decide(parent_observer_sha256="same", current_observer_sha256="same", requested=True)
    assert not decide(parent_observer_sha256="old", current_observer_sha256="new", requested=False)


def test_bilateral_trace_annotation_uses_bound_holding_predicate():
    controller = {"predicate_diagnostics": [{"predicate": "holding", "value": True}]}
    assert runner.bilateral_holding_annotation({"holding_contract": "bilateral_contact"}, controller)
    assert not runner.bilateral_holding_annotation({"holding_contract": "loaded_contact"}, controller)
    assert not runner.bilateral_holding_annotation(
        {"holding_contract": "bilateral_contact"},
        {"predicate_diagnostics": [{"predicate": "holding", "value": None}]})


def test_stage_goal_contract_is_explicit_bounded_and_private():
    rows = graph._predicate_rows(
        [{"predicate": "stage_goal", "key": "goal_inside", "debounce_sec": .15}],
        label="completion")
    assert rows[0]["key"] == "goal_inside" and rows[0]["debounce_sec"] == .15
    for malformed in (
        {"predicate": "stage_goal"},
        {"predicate": "stage_goal", "key": "goal_inside", "debounce_sec": .151},
    ):
        try:
            graph._predicate_rows([malformed], label="completion")
        except ValueError:
            pass
        else:
            raise AssertionError("malformed stage_goal contract was accepted")
    stage = {"object": "apple", "target": "bowl", "goal_relation": "inside",
             "holding_contract": "bilateral_contact"}
    predicate = rows[0]
    common = dict(stage=stage, progress=1.0, raw={"goal_inside": True},
                  object_value=None, target_value=None, tcp=np.zeros(3),
                  relation_reference={}, entry_geometry={})
    value, diagnostic = runner.evaluate_contract_predicate(predicate, **common)
    assert value is None  # Visible/raw aliases cannot satisfy the private gate.
    value, diagnostic = runner.evaluate_contract_predicate(
        predicate, stage_goal_raw={predicate["id"]: True}, **common)
    assert value is True and diagnostic["authority"] == "private_boolean_stage_supervisor"


def test_stage_goal_rows_dispatch_independent_roles_relations_and_debounce():
    rows = graph._predicate_rows([
        {"predicate": "stage_goal", "key": "goal_lifted",
         "goal_relation": "lifted", "debounce_sec": .03},
        {"predicate": "stage_goal", "key": "goal_sweep_inside",
         "object_role": "yellow_cube", "target_role": "dustpan",
         "goal_relation": "block_inside_dustpan_with_tools_held", "debounce_sec": .14},
    ], label="completion")
    class World:
        def __init__(self): self.calls = []
        def stage_goal_facts(self, object_role, target_role, **kwargs):
            self.calls.append((object_role, target_role, kwargs))
            return {"goal_lifted": True, "goal_sweep_inside": False}
    world = World()
    stage = {"object": "broom", "target": "yellow_cube", "arm": "right",
             "goal_relation": "inside"}
    values = runner.query_stage_goal_predicates(world, stage, rows)
    assert values == {rows[0]["id"]: True, rows[1]["id"]: False}
    assert world.calls[0] == ("broom", "yellow_cube", {
        "arm": "right", "goal_relation": "lifted", "debounce_sec": .03})
    assert world.calls[1] == ("yellow_cube", "dustpan", {
        "arm": "right", "goal_relation": "block_inside_dustpan_with_tools_held",
        "debounce_sec": .14})


def test_actor_command_reached_uses_post_step_active_arm_error():
    rows = graph._predicate_rows(
        [{"predicate": "actor_command_reached"}], label="completion")
    predicate = rows[0]
    assert predicate["tolerance_rad"] == .025
    stage = {"object": "apple", "target": "bowl", "goal_relation": "inside",
             "holding_contract": "bilateral_contact"}
    common = dict(stage=stage, progress=1.0, raw={}, object_value=None,
                  target_value=None, tcp=np.zeros(3), relation_reference={},
                  entry_geometry={})
    value, diagnostic = runner.evaluate_contract_predicate(predicate, **common)
    assert value is None  # Reference progress cannot stand in for joint feedback.
    value, diagnostic = runner.evaluate_contract_predicate(
        predicate, actor_joint_error_max_rad=.0249, **common)
    assert value is True and diagnostic["error_max_rad"] == .0249
    value, _ = runner.evaluate_contract_predicate(
        predicate, actor_joint_error_max_rad=.0251, **common)
    assert value is False
    for tolerance in (0.0, .1001):
        try:
            graph._predicate_rows(
                [{"predicate": "actor_command_reached", "tolerance_rad": tolerance}],
                label="completion")
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range actor tolerance was accepted")


def test_r2_attachment_exact_huber_gradient_and_loss():
    latent = skill.AttachmentPoseLatent(enabled=True)
    b = np.array([.010, -.002, .001])
    record = latent.update(b, admitted=True, admission_reason="admitted",
                           matched_reference_time_sec=1.2)
    expected_gradient = np.clip(-b, -.004, .004)
    expected_z = -(.25 / 1.15) * expected_gradient
    np.testing.assert_allclose(record["gradient_local"], expected_gradient)
    np.testing.assert_allclose(latent.z, expected_z)
    error = -b
    expected_loss = np.sum(np.where(
        np.abs(error) <= .004, .5 * error ** 2,
        .004 * (np.abs(error) - .002)))
    assert np.isclose(record["loss_before"], expected_loss)
    assert record["loss_after"] < record["loss_before"]


def test_r2_feedback_admission_rotation_and_action_effect(tmp_path):
    input_dim, output_dim = 29, 6
    metadata = {"schema": skill.FrozenNumpyMLP.schema, "z_feature_slice": [26, 29],
                "correction_feature_slices": [[12, 18], [26, 29]], "action_limit_rad": 1.0}
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "input_mean": np.zeros(input_dim), "input_scale": np.ones(input_dim),
        "output_mean": np.zeros(output_dim), "output_scale": np.ones(output_dim),
        "w0": np.zeros((64, input_dim)), "b0": np.zeros(64),
        "w1": np.zeros((128, 64)), "b1": np.zeros(128),
        "w2": np.zeros((output_dim, 128)), "b2": np.zeros(output_dim),
    }
    arrays["w0"][0, 26:29] = -1.0
    arrays["w1"][0, 0] = 1.0; arrays["w2"][0, 0] = 1.0
    checkpoint = tmp_path / "r2_skill.npz"; np.savez(checkpoint, **arrays)
    contract = json.loads((ROOT / "evaluations/twin_rsi_v01/R2_FEEDBACK_CONTRACT.json").read_text())
    runtime = skill.TwinLocalSkill(checkpoint, online_z=True, feedback_contract=contract)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    kwargs = dict(phase="Align", holding=True, matched_reference_time_sec=2.0,
                  current_tcp_xyz=np.zeros(3), current_tcp_rotation=rotation,
                  reference_tcp_xyz=np.zeros(3), reference_tcp_rotation=np.eye(3),
                  reference_surface_xyz=np.zeros(3))
    first = runtime.command(np.zeros(input_dim), current_surface_xyz=[0, .01, 0], **kwargs)
    assert first.record["latent"]["admission_reason"] == "first_local_action_common_prefix"
    assert first.record["latent"]["update_index"] == 0
    missing = runtime.command(np.zeros(input_dim), current_surface_xyz=None, **kwargs)
    assert missing.record["latent"]["admission_reason"] == "fresh_visible_object_missing"
    assert missing.record["latent"]["update_index"] == 0
    not_held = runtime.command(
        np.zeros(input_dim), current_surface_xyz=[0, .01, 0],
        **{**kwargs, "holding": False})
    assert not_held.record["latent"]["admission_reason"] == "current_holding_not_true"
    assert not_held.record["latent"]["update_index"] == 0
    held = runtime.command(np.zeros(input_dim), current_surface_xyz=[0, .01, 0], **kwargs)
    assert held.record["latent"]["admitted"] and held.record["latent"]["update_index"] == 1
    z_local = np.asarray(held.record["latent"]["z_local_after_m"])
    z_world = np.asarray(held.record["latent"]["z_world_m"])
    np.testing.assert_allclose(z_world, -rotation @ z_local)
    np.testing.assert_allclose(held.record["latent"]["z_after_m"], z_world)
    assert held.record["action_online_minus_fixed_l2_rad"] > 0
    runtime.reset_phase(completed_stage_id="c01.05.align")
    assert runtime.phase_archive[-1]["updates"] == 1
    assert runtime.phase_archive[-1]["stage_id"] == "c01.05.align"
    fixed = skill.TwinLocalSkill(checkpoint, online_z=False, feedback_contract=contract)
    fixed.command(np.zeros(input_dim), current_surface_xyz=[0, .01, 0], **kwargs)
    observed = fixed.command(np.zeros(input_dim), current_surface_xyz=[0, .01, 0], **kwargs)
    assert observed.record["latent"]["admitted"]
    assert observed.record["latent"]["attachment_discrepancy_local_m"] is not None
    assert observed.record["latent"]["update_index"] == 0
    np.testing.assert_allclose(observed.record["latent"]["z_world_m"], np.zeros(3))


def test_r2_feedback_requires_bilateral_not_supported_contact():
    raw = {"bilateral_contact": False, "object_support_contact": True}
    assert runner.holding_fact("held_or_supported", raw) is True
    assert runner.attachment_feedback_holding(raw) is False


def test_compiler_copies_feedback_contract_and_merges_entry_override():
    plan = json.loads((ROOT / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json").read_text())
    revision = json.loads((ROOT / "evaluations/twin_rsi_v01/R1_OVERRIDES.json").read_text())
    feedback = json.loads((ROOT / "evaluations/twin_rsi_v01/R2_FEEDBACK_CONTRACT.json").read_text())
    revision["local_feedback_contract"] = feedback
    apple_revision = next(row for row in revision["tasks"] if row["task"] == "collect_apple")
    apple_revision["cycles"][0]["source_anchor_role"] = "bowl"
    wait = next(row for row in apple_revision["cycles"][0]["stage_contracts"] if row["phase"] == "Wait")
    wait["entry"] = {"role_visible_or_held": False}
    task = next(row for row in plan["tasks"] if row["task"] == "collect_apple")
    with np.load(task["sources"]["motor_targets.npz"]["path"], allow_pickle=False) as data:
        times = data["timestamps"]
    compiled = graph.compile_task_graph(
        task, root_plan_sha256="test", reference_times=times, revision=revision,
        revision_sha256="test-r2")
    assert compiled["local_feedback_contract"] == feedback
    assert all(row["source_anchor_role"] == "bowl" for row in compiled["stages"])
    compiled_wait = next(row for row in compiled["stages"] if row["phase"] == "Wait")
    assert compiled_wait["entry"]["role_visible_or_held"] is False


def test_test_selection_binds_unique_task_artifacts_and_relative_snapshot(tmp_path):
    solution = tmp_path / "tasks/collect_apple/solution"
    for relative, content in (
        ("graph/task_graph.json", b"graph"),
        ("models/local_skill.npz", b"model"),
        ("refs/visible_centroids.json", b"refs"),
        ("CANDIDATE_FREEZE.json", b"freeze"),
    ):
        path = solution / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    snapshot = tmp_path / "source_snapshots/source-a"; snapshot.mkdir(parents=True)
    selection_path = tmp_path / "FINAL_SELECTION.json"
    row = {
        "task": "collect_apple", "candidate_id": "r2",
        "packages": "tasks/collect_apple/solution",
        "source_snapshot": "source_snapshots/source-a",
        "graph_sha256": runner.digest(solution / "graph/task_graph.json"),
        "model_sha256": runner.digest(solution / "models/local_skill.npz"),
        "reference_sha256": runner.digest(solution / "refs/visible_centroids.json"),
        "candidate_freeze_sha256": runner.digest(solution / "CANDIDATE_FREEZE.json"),
    }
    selection = {"status": "frozen", "candidate_id": "per_task_selected_R0_to_R3",
                 "tasks": [row]}
    assert runner.validate_test_selection(
        selection, task="collect_apple", candidate_id="r2", solution=solution,
        selection_path=selection_path) == row
    bad = json.loads(json.dumps(selection)); bad["tasks"][0]["model_sha256"] = "bad"
    with pytest.raises(ValueError, match="artifact binding mismatch"):
        runner.validate_test_selection(
            bad, task="collect_apple", candidate_id="r2", solution=solution,
            selection_path=selection_path)
    duplicate = {**selection, "tasks": [row, dict(row)]}
    with pytest.raises(ValueError, match="exactly one"):
        runner.validate_test_selection(
            duplicate, task="collect_apple", candidate_id="r2", solution=solution,
            selection_path=selection_path)


def test_r3_initial_grounding_and_bounded_reference_pairing():
    frame = {"physical_time_sec": 2.0, "roles": {
        "apple": {"centroid_world_m": [.11, .2, .3]},
        "hidden": {"centroid_world_m": None},
    }}
    reference = {"time_sec": 0.0, "surface_centroids_world_m": {
        "apple": [.10, .18, .3], "hidden": [0, 0, 0]}}
    priors = runner.scene_entry_grounding_priors(frame, reference, ["apple", "hidden"])
    np.testing.assert_allclose(priors["apple"]["delta_m"], [.01, .02, 0])
    assert priors["apple"]["fresh_feedback"] is False and "hidden" not in priors
    stage = {"object": "apple", "source_grounding": "initial_visible_until_first_manipulation"}
    assert runner.use_scene_entry_grounding(stage, set())
    assert not runner.use_scene_entry_grounding(stage, {"apple"})
    cache = runner.ReferenceCentroidCache(.5)
    value, state = cache.get("apple", {"apple": [1, 2, 3]}, 1.0)
    assert state["fresh"]
    value, state = cache.get("apple", {"apple": None}, 1.5)
    np.testing.assert_allclose(value, [1, 2, 3]); assert state["held_previous"]
    value, state = cache.get("apple", {"apple": None}, 1.501)
    assert value is None and state["reason"] == "past_reference_expired"


def test_r3_support_holding_is_bound_to_declared_target():
    raw = {"bilateral_contact": False, "object_support_contact": True,
           "object_table_support_contact": False}
    assert runner.holding_fact("held_or_supported", raw, target_role="board") is True
    assert runner.holding_fact(
        "held_or_supported", raw, target_role="table_handover_region") is False
    raw["object_table_support_contact"] = True
    assert runner.holding_fact(
        "held_or_supported", raw, target_role="table_handover_region",
        goal_relation="accessible_to_right_arm") is True
    assert runner.holding_fact(
        "held_or_supported", raw, target_role="table_handover_region",
        goal_relation="inside") is False


def test_v02_fixed_support_projection_is_explicit_and_cycle_scoped():
    projected, record = runner.project_source_grounding(
        [.023, -.011, .014], "fixed_support_xy_z0")
    np.testing.assert_allclose(projected, [.023, -.011, 0.0])
    assert record["raw_delta_m"] == [.023, -.011, .014]
    assert record["projected"] and record["uses_continuous_object_truth"] is False
    unchanged, legacy = runner.project_source_grounding([.01, .02, .03], "none")
    np.testing.assert_allclose(unchanged, [.01, .02, .03])
    assert not legacy["projected"]

    plan = json.loads((ROOT / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json").read_text())
    revision = json.loads((ROOT / "evaluations/twin_rsi_v01/R3_OVERRIDES.json").read_text())
    task_revision = next(x for x in revision["tasks"] if x["task"] == "put_mug_on_box")
    cycle = next(x for x in task_revision["cycles"] if x["cycle_id"] == "c02")
    cycle["source_grounding_plane"] = "fixed_support_xy_z0"
    task = next(x for x in plan["tasks"] if x["task"] == "put_mug_on_box")
    with np.load(task["sources"]["motor_targets.npz"]["path"], allow_pickle=False) as data:
        compiled = graph.compile_task_graph(
            task, root_plan_sha256="test", reference_times=data["timestamps"], revision=revision)
    assert {x["source_grounding_plane"] for x in compiled["stages"] if x["cycle_id"] == "c02"} == {
        "fixed_support_xy_z0"}
    assert {x["source_grounding_plane"] for x in compiled["stages"] if x["cycle_id"] == "c01"} == {"none"}


def test_v02_runtime_contract_and_coherent_knife_condition_are_explicit_opt_in():
    contract = runner.validate_runtime_contract({
        "schema": runner.V02_RUNTIME_SCHEMA,
        "cup_release_control": "portable_tcp_displacement_v1",
        "numerical_integrity_fail_closed": True,
        "condition_design": {"knife_fixture_group_translation": True},
    })
    plan = json.loads((ROOT / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json").read_text())
    knife = next(x for x in plan["tasks"] if x["task"] == "put_knife_board_new")
    condition = runner.condition_manifest(
        knife, seed=123, trial=4, nominal=False,
        condition_design=contract["condition_design"])
    shifts = condition["actual_initial_translation_m"]
    assert shifts["knife_holder"] == shifts["red_knife"] == shifts["blue_knife"]
    assert shifts["board"] != shifts["knife_holder"]
    assert "ood" not in condition["family"]
    legacy = runner.condition_manifest(knife, seed=123, trial=4, nominal=False)
    assert legacy["family"] == "table_plane_position_ood_3_to_8mm"
    assert "condition_design" not in legacy
    with pytest.raises(ValueError, match="portable"):
        runner.validate_runtime_contract({
            "schema": runner.V02_RUNTIME_SCHEMA,
            "cup_release_control": "legacy_object_distance",
            "numerical_integrity_fail_closed": True,
            "condition_design": {"knife_fixture_group_translation": True},
        })


def test_r3_compiler_and_training_cli_are_explicit_opt_in():
    parsed = runner.parser().parse_args([
        "prepare", "--candidate-id", "r3", "--task", "collect_apple",
        "--samples-per-stage", "4096", "--qpos-jitter-rad", ".015",
        "--overrides", "r3.json"])
    assert parsed.samples_per_stage == 4096 and parsed.qpos_jitter_rad == .015
    assert parsed.overrides == Path("r3.json")
    trained = runner.parser().parse_args([
        "train", "--candidate-id", "r3", "--task", "collect_apple", "--steps", "10000",
        "--action-limit-rad", ".08"])
    assert trained.steps == 10000 and trained.action_limit_rad == .08
    plan = json.loads((ROOT / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json").read_text())
    revision = json.loads((ROOT / "evaluations/twin_rsi_v01/R1_OVERRIDES.json").read_text())
    revision["source_grounding"] = "initial_visible_until_first_manipulation"
    apple_revision = next(row for row in revision["tasks"] if row["task"] == "collect_apple")
    task = next(row for row in plan["tasks"] if row["task"] == "collect_apple")
    with np.load(task["sources"]["motor_targets.npz"]["path"], allow_pickle=False) as data:
        compiled = graph.compile_task_graph(
            task, root_plan_sha256="test", reference_times=data["timestamps"], revision=revision)
    assert all(stage["source_grounding"] == "initial_visible_until_first_manipulation"
               for stage in compiled["stages"])
    move = next(row for row in apple_revision["cycles"][0]["stage_contracts"]
                if row["phase"] == "Move")
    move["holding_contract"] = "held_or_supported"
    with np.load(task["sources"]["motor_targets.npz"]["path"], allow_pickle=False) as data:
        with pytest.raises(ValueError, match="restricted"):
            graph.compile_task_graph(
                task, root_plan_sha256="test", reference_times=data["timestamps"], revision=revision)
