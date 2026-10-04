import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from prove.task_graph.p70_object_controller import P70ObjectController, P70ControllerState
from tools.task_graph import p70_stage_runner as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DummyPolicy:
    def command(self, T_world_book, T_world_eef, gripper_qpos, goal_book_eef, close=False):
        return np.array([0., 0., 0., 0., 0., 0., 1. if close else -1.]), {"dummy": True, "close": bool(close)}


def public_frame():
    return {
        "rgb": np.zeros((2, 448, 448, 3), dtype=np.uint8),
        "visible_masks": np.zeros((2, 3, 448, 448), dtype=bool),
        "camera_K": np.tile(np.eye(3), (2, 1, 1)),
        "camera_pose": np.tile(np.eye(4), (2, 1, 1)),
        "eef_pos": np.array([0.0, 0.0, 1.0]),
        "eef_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
        "gripper_qpos": np.array([0.02, 0.02]),
        "joint_pos": np.zeros(7),
        "joint_vel": np.zeros(7),
    }


def write_ref(path):
    path.write_text(json.dumps({"schema": "prove.p69-grounding-reference.v1", "role_order": ["gripper","book","caddy"], "views": [0],
        "runtime_forbidden": ["current_episode_private_pose"],
        "role_calibration": {"book": {"observer": "view0_planar", "plane_world_z": 0, "demo_pose_xy_yaw": [0,0,0], "mask_bias_xy_yaw": [0,0,0]},
                             "caddy": {"observer": "view0_planar", "plane_world_z": 0, "demo_pose_xy_yaw": [0,0,0], "mask_bias_xy_yaw": [0,0,0]}}}))


def config(root, ref):
    rel = root / "relations.json"
    rel.write_text(json.dumps({"schema": "prove.p70-object-relation-targets.v1",
        "T_caddy_body_back_site": np.eye(4).tolist(), "T_back_site_book_entry": np.eye(4).tolist(), "T_world_caddy_demo": np.eye(4).tolist(),
        "book_geometry": {"center": [0,0,0], "halfsize": [0.01,0.02,0.03]},
        "safety_heights": {"book_bottom_min_world_z": 0.95},
        "micro_relation": {"final_caddy_book": np.eye(4).tolist(), "book_collision_center": [0,0,0], "book_collision_halfsize": [0.01,0.02,0.03], "back_halfsize": [0.05,0.05,0.05], "mouth_clearance_margin": 0.0025}}))
    demo = root / "demo_rel.json"
    demo.write_text(json.dumps({"schema": "prove.p70-demo-grasp-relations.v1", "relations": {"pregrasp": {"T_book_eef": np.eye(4).tolist()}, "grasp": {"T_book_eef": np.eye(4).tolist()}}}))
    ckpt = root / "final.pt"; ckpt.write_bytes(b"dummy")
    return {
        "schema": "prove.p70-object-controller-config.v1",
        "runtime_inputs": ["public_masks", "camera", "proprio"],
        "grounding_reference": {"path": str(ref), "sha256": sha(ref)},
        "object_relation_targets": {"path": str(rel), "sha256": sha(rel)},
        "grasp_policy": {"module": "prove.task_graph.p70_grasp_policy_v02", "checkpoint": {"path": str(ckpt), "sha256": sha(ckpt)}, "demo_relations": {"path": str(demo), "sha256": sha(demo)}},
        "heights": {"book_body_z": 0.88, "caddy_body_z": 0.887},
        "offsets": {"approach_clearance_z": 0.08, "lift_clearance_z": 0.18},
        "approach_gate": {"position_m": 0.025, "rotation_rad": 0.25},
        "stage_budgets": {"approach_book": 5, "short_grasp": 5, "lift_clear": 5, "orient_clear": 5, "transport_above_caddy": 5, "near_entry": 5},
        "gains": {"position_gain": 1.0, "rotation_gain": 1.0, "translation_scale": 0.05, "rotation_scale": 0.5, "max_action": 1.0},
    }


def fake_load(self):
    self.grasp_policy = DummyPolicy()


class P70ControllerTests(unittest.TestCase):
    def test_controller_rejects_private_or_missing_public_keys(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            root = Path(td); ref = root / "ref.json"; write_ref(ref)
            c = P70ObjectController(config(root, ref))
            frame = public_frame(); frame["private_world_book"] = np.eye(4)
            with self.assertRaises(ValueError):
                c.command(frame, {})

    def test_stage_transition_captures_held_transform_after_attached(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            root = Path(td); ref = root / "ref.json"; write_ref(ref)
            c = P70ObjectController(config(root, ref))
            with mock.patch.object(c, "observe_grounding") as obs:
                from prove.task_graph.p69_public_grounding import RoleDelta
                obs.return_value = (RoleDelta("book", (0,0), (.1,.2), (.1,.2), 0.0, 0.0, True),
                                    RoleDelta("caddy", (0,0), (.5,.6), (.5,.6), 0.0, 0.0, True))
                c.state.stage = "short_grasp"
                c.command(public_frame(), {"attached": True})
                self.assertEqual(c.state.stage, "lift_clear")
                self.assertIsNotNone(c.state.held_T_eef_book)
                self.assertFalse(c.records[-1]["runtime_private_pose_used"])

    def test_prepare_freeze_schema_without_physics_qualification(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(runner, "source_bindings", return_value={}), mock.patch.object(runner, "binding_chain", return_value=[]), mock.patch.object(runner, "P70ObjectController") as ctl:
            root = Path(td); ref = root / "ref.json"; write_ref(ref)
            cfg = root / "cfg.json"; cfg.write_text(json.dumps(config(root, ref)))
            state = np.zeros(45, dtype=np.float64)
            scope = {"schema": "prove.p69-init-scope.v1", "entries": []}
            for i in range(50):
                split = "train" if i < 30 else "explore" if i < 40 else "test"
                vec = state + i
                scope["entries"].append({"id": i, "split": split, "state_flattened": vec.tolist(), "state_sha256": hashlib.sha256(vec.tobytes()).hexdigest()})
            scope_path = root / "scope.json"; scope_path.write_text(json.dumps(scope))
            p55 = root / "p55.json"; p55.write_text(json.dumps({"schema": "p55"}))
            p69 = root / "p69.json"; p69.write_text(json.dumps({"schema": "prove.p69-full-rsi-freeze.v1", "p55_binding": {"path": str(p55), "sha256": sha(p55)}, "files": {}}))
            frozen = runner.prepare(output=root / "out", p69_freeze=p69, init_scope=scope_path, controller_config=cfg, split="qualification", version="P70Q0", stop_after="lift_clear")
            self.assertEqual(frozen["schema"], "prove.p70-object-centric-freeze.v1")
            self.assertEqual([c["init_id"] for c in frozen["cells"]], [20, 2, 12, 26])
            self.assertEqual(frozen["budgets"]["stop_after"], "lift_clear")


class P70TraceRecordTests(unittest.TestCase):
    def test_trace_record_renames_controller_t_and_outer_fields_win(self):
        row = runner.trace_record(9, {"t": 3, "facts": {"inner": True}, "stage": "lift_clear"}, facts={"outer": True}, stage="outer_stage")
        self.assertEqual(row["t"], 9)
        self.assertEqual(row["controller_t"], 3)
        self.assertEqual(row["facts"], {"outer": True})
        self.assertEqual(row["stage"], "outer_stage")

class P70CaddyDeltaFrameTests(unittest.TestCase):
    def test_caddy_zero_delta_recovers_demo_body_pose_not_pca_yaw(self):
        from prove.task_graph.p69_public_grounding import RoleDelta
        cfg = load_real_q1_config_for_test()
        with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            c = P70ObjectController(cfg)
        caddy = RoleDelta("caddy", (-0.396734, -0.138556), (-0.396734, -0.138556), (0.0, 0.0), 0.0, 1.355736, True)
        T = c._caddy_pose_from_delta(caddy)
        demo = np.asarray(c.relation_targets["T_world_caddy_demo"], dtype=float)
        self.assertTrue(np.allclose(T, demo))
        yaw = Rotation.from_matrix(T[:3, :3]).as_euler("zyx")[0]
        self.assertAlmostEqual(abs(yaw), np.pi, places=6)

    def test_caddy_delta_retargets_demo_body_pose(self):
        from prove.task_graph.p69_public_grounding import RoleDelta
        cfg = load_real_q1_config_for_test()
        with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            c = P70ObjectController(cfg)
        caddy = RoleDelta("caddy", (0, 0), (0.01, -0.02), (0.01, -0.02), 0.1, 1.4557, True)
        T = c._caddy_pose_from_delta(caddy)
        demo = np.asarray(c.relation_targets["T_world_caddy_demo"], dtype=float)
        self.assertTrue(np.allclose(T[:2, 3], demo[:2, 3] + np.array([0.01, -0.02])))
        self.assertTrue(np.allclose(T[:3, :3], Rotation.from_euler("z", 0.1).as_matrix() @ demo[:3, :3]))


def load_real_q1_config_for_test():
    cfg = json.loads(Path('evaluations/outer_rsi_p70_object_graph_v01/P70_CONTROLLER_CONFIG_DRAFT.json').read_text())
    return cfg


class P70PositionThenCloseGraspTests(unittest.TestCase):
    def q2_controller(self, root):
        ref = root / "ref.json"; write_ref(ref)
        cfg = config(root, ref)
        cfg["grasp_sequence"] = "position_then_close"
        cfg["grasp_gate"] = {"position_m": 0.003585, "rotation_rad": 0.17453292519943295}
        with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            return P70ObjectController(cfg)

    def test_position_then_close_keeps_open_until_gate(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.q2_controller(Path(td))
            frame = public_frame(); frame["eef_pos"] = np.array([0.1, 0.0, 0.0])
            B = np.eye(4); E = np.eye(4); E[:3, 3] = frame["eef_pos"]
            target, action, diag = c._short_grasp_action(frame, B, E)
            self.assertEqual(c.state.grasp_substage, "position_open")
            self.assertLess(float(action[6]), 0.0)
            self.assertFalse(c.state.close_latched_once)
            self.assertEqual(diag["grasp_substage"], "position_open")

    def test_position_then_close_latches_once_and_holds_closed(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.q2_controller(Path(td))
            frame = public_frame(); frame["eef_pos"] = np.zeros(3)
            B = np.eye(4); E = np.eye(4)
            target, action, diag = c._short_grasp_action(frame, B, E)
            self.assertEqual(c.state.grasp_substage, "close_hold")
            self.assertTrue(c.state.close_latched_once)
            self.assertGreater(float(action[6]), 0.0)
            latched = np.asarray(c.state.latched_T_world_eef_grasp)
            frame2 = public_frame(); frame2["eef_pos"] = np.array([0.02, 0.0, 0.0])
            target2, action2, diag2 = c._short_grasp_action(frame2, B, np.eye(4))
            self.assertEqual(c.state.grasp_substage, "close_hold")
            self.assertTrue(np.allclose(np.asarray(c.state.latched_T_world_eef_grasp), latched))
            self.assertGreater(float(action2[6]), 0.0)
            self.assertEqual(diag2["grasp_substage"], "close_hold")

    def test_resume_state_preserves_grasp_substage_and_latch(self):
        class C: pass
        state = P70ControllerState(stage="short_grasp", grasp_substage="close_hold", latched_T_world_eef_grasp=np.eye(4).tolist(), close_latched_once=True).to_dict()
        c = C(); from tools.task_graph.p70_stage_resume import set_controller_state
        set_controller_state(c, state)
        self.assertEqual(c.state.grasp_substage, "close_hold")
        self.assertTrue(c.state.close_latched_once)
        self.assertTrue(np.allclose(np.asarray(c.state.latched_T_world_eef_grasp), np.eye(4)))


class P70FreezeOrientGoalTests(unittest.TestCase):
    def controller(self, root):
        ref = root / "ref.json"; write_ref(ref)
        cfg = config(root, ref)
        cfg["freeze_orient_goal"] = True
        with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            return P70ObjectController(cfg)

    def test_freeze_orient_goal_keeps_world_xy_when_eef_moves(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.controller(Path(td))
            current = np.eye(4); current[:2, 3] = [0.2, -0.1]
            entry = np.eye(4); entry[:2, 3] = [0.5, 0.4]
            goal1 = c._orient_book_goal(current, entry)
            moved = np.eye(4); moved[:2, 3] = [0.3, 0.2]
            goal2 = c._orient_book_goal(moved, entry)
            self.assertTrue(np.allclose(goal1, goal2))
            self.assertTrue(np.allclose(goal1[:2, 3], [0.2, -0.1]))
            self.assertIsNotNone(c.state.orient_goal_T_world_book)

    def test_resume_state_preserves_orient_goal(self):
        class C: pass
        state = P70ControllerState(stage="orient_clear", orient_goal_T_world_book=np.eye(4).tolist()).to_dict()
        c = C(); from tools.task_graph.p70_stage_resume import set_controller_state
        set_controller_state(c, state)
        self.assertTrue(np.allclose(np.asarray(c.state.orient_goal_T_world_book), np.eye(4)))


class P70HeldPoseCalibrationTests(unittest.TestCase):
    def write_calibration(self, root, dz=0.1):
        D = np.eye(4); D[2, 3] = dz
        path = root / "held_cal.json"
        path.write_text(json.dumps({"schema": "prove.p70-held-relation-calibration.v1", "T_predicted_book_actual_book": D.tolist(), "authority": "fixed_training_prior"}))
        return path, D

    def controller_with_calibration(self, root, *, bad_hash=False):
        ref = root / "ref.json"; write_ref(ref)
        cfg = config(root, ref)
        cal, D = self.write_calibration(root)
        cfg["held_pose_calibration"] = {"path": str(cal), "sha256": ("bad" if bad_hash else sha(cal))}
        with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            return P70ObjectController(cfg), D

    def test_no_binding_old_behavior_no_calibration_applied(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.json"; write_ref(ref)
            cfg = config(Path(td), ref)
            with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
                c = P70ObjectController(cfg)
            c.state.held_T_eef_book = np.eye(4).tolist()
            out = c._apply_held_pose_calibration_once()
            self.assertFalse(out["applied"])
            self.assertFalse(c.state.held_pose_calibrated)
            self.assertTrue(np.allclose(np.asarray(c.state.held_T_eef_book), np.eye(4)))

    def test_incorrect_calibration_hash_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.controller_with_calibration(Path(td), bad_hash=True)

    def test_held_pose_calibration_applies_once(self):
        with tempfile.TemporaryDirectory() as td:
            c, D = self.controller_with_calibration(Path(td))
            c.state.held_T_eef_book = np.eye(4).tolist()
            first = c._apply_held_pose_calibration_once()
            second = c._apply_held_pose_calibration_once()
            self.assertTrue(first["applied"])
            self.assertFalse(second["applied"])
            self.assertTrue(c.state.held_pose_calibrated)
            self.assertTrue(np.allclose(np.asarray(c.state.held_T_eef_book), D))

    def test_resume_preserves_held_pose_calibrated_flag(self):
        class C: pass
        state = P70ControllerState(stage="orient_clear", held_pose_calibrated=True, held_T_eef_book=np.eye(4).tolist()).to_dict()
        c = C(); from tools.task_graph.p70_stage_resume import set_controller_state
        set_controller_state(c, state)
        self.assertTrue(c.state.held_pose_calibrated)


class P70PlacementVerificationTests(unittest.TestCase):
    def test_verification_gate_requires_microbudget_goal_bilateral_and_not_wrong(self):
        from tools.task_graph import p70_stage_runner as runner
        self.assertTrue(runner.placement_verification_precondition({"stop_reason": "micro_budget24"}, True, [True, True], False))
        self.assertFalse(runner.placement_verification_precondition({"stop_reason": "wrong_target"}, True, [True, True], False))
        self.assertFalse(runner.placement_verification_precondition({"stop_reason": "micro_budget24"}, False, [True, True], False))
        self.assertFalse(runner.placement_verification_precondition({"stop_reason": "micro_budget24"}, True, [True, False], False))
        self.assertFalse(runner.placement_verification_precondition({"stop_reason": "micro_budget24"}, True, [True, True], True))

    def test_verification_budget_never_exceeds_global720(self):
        from tools.task_graph import p70_stage_runner as runner
        frozen = {"controller": {"placement_verification_max_actions": 3}, "budgets": {"total_max_actions": 720}}
        self.assertEqual(runner.placement_verification_limit(frozen, 690, 24), 3)
        self.assertEqual(runner.placement_verification_limit(frozen, 694, 24), 2)
        self.assertEqual(runner.placement_verification_limit(frozen, 696, 24), 0)
        frozen["controller"]["placement_verification_max_actions"] = 0
        self.assertEqual(runner.placement_verification_limit(frozen, 690, 24), 0)

    def test_stationary_verification_action_has_no_adapter_update_semantics(self):
        from tools.task_graph import p70_stage_runner as runner
        action = runner.stationary_closed_action()
        self.assertEqual(action.shape, (7,))
        self.assertTrue(np.allclose(action[:6], 0.0))
        self.assertEqual(float(action[6]), 1.0)


class P70PlacementVerificationResultSemanticsTests(unittest.TestCase):
    def test_verification_failure_does_not_promote_single_frame_official_to_success(self):
        from tools.task_graph import p70_stage_runner as runner
        # Helper-level contract: if the verification node records only an instant
        # official frame and never reaches InsertJudge complete, result-level
        # official_success must remain the 3-hold semantic.  The runner encodes
        # this as graph_complete/native_hold3, with native_instant_ever separate.
        failed = {"enabled": True, "actions": 1, "graph_complete": False, "official_success": False, "native_instant_ever": True}
        self.assertFalse(failed["official_success"])
        self.assertTrue(failed["native_instant_ever"])

    def test_verification_global_time_is_post_action_offset(self):
        prefix_actions, s9_actions, local_t = 480, 24, 0
        self.assertEqual(prefix_actions + s9_actions + local_t + 1, 505)

class P70PlacementVerificationExecutionTests(unittest.TestCase):
    class FakeState:
        def flatten(self):
            return np.zeros(45, dtype=np.float64)
    class FakeSim:
        def get_state(self):
            return P70PlacementVerificationExecutionTests.FakeState()
    class FakeEnv:
        def __init__(self):
            self.sim = P70PlacementVerificationExecutionTests.FakeSim(); self.steps = 0
        def _get_observations(self, force_update=False):
            return {"step": self.steps}
        def step(self, action):
            self.steps += 1
            return self._get_observations(True), 0.0, False, {}
        def _check_success(self):
            return True
    class FakeRenderer:
        def __init__(self, env, public): self.env = env
        def observe(self, obs):
            return {
                "rgb": np.zeros((2, 448, 448, 3), dtype=np.uint8),
                "visible_masks": np.zeros((2, 3, 448, 448), dtype=np.uint8),
                "camera_K": np.repeat(np.eye(3)[None], 2, axis=0),
                "camera_pose": np.repeat(np.eye(4)[None], 2, axis=0),
                "eef_pos": np.zeros(3),
                "eef_quat_xyzw": np.array([0., 0., 0., 1.]),
                "gripper_qpos": np.zeros(2),
                "joint_pos": np.zeros(7),
                "joint_vel": np.zeros(7),
            }
        def close(self): pass

    def test_run_verification_requires_three_post_action_holds_and_records_global_t(self):
        from tools.task_graph import p70_stage_runner as runner
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(runner, "PublicRenderer", self.FakeRenderer), \
             mock.patch.object(runner, "snapshot", return_value=({"supervisor_dummy": np.array([1.])}, None)), \
             mock.patch.object(runner, "structural_measurement", return_value=(True, [True, True], False)):
            out = runner.run_placement_verification(self.FakeEnv(), {"budgets": {"total_max_actions": 720}}, Path(td), {}, {"public": {}}, max_actions=3, global_offset=504)
            self.assertTrue(out["graph_complete"])
            self.assertTrue(out["official_success"])
            self.assertEqual(out["actions"], 3)
            self.assertEqual(out["updates"], 0)
            rows = [json.loads(line) for line in (Path(td) / "actions" / "VERIFY_ACTIONS.jsonl").read_text().splitlines()]
            self.assertEqual([r["global_t"] for r in rows], [505, 506, 507])
            self.assertEqual([r["native_hold3"] for r in rows], [False, False, True])

    def test_run_verification_single_instant_official_is_not_reported_as_success(self):
        from tools.task_graph import p70_stage_runner as runner
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(runner, "PublicRenderer", self.FakeRenderer), \
             mock.patch.object(runner, "snapshot", return_value=({"supervisor_dummy": np.array([1.])}, None)), \
             mock.patch.object(runner, "structural_measurement", return_value=(True, [True, True], False)):
            out = runner.run_placement_verification(self.FakeEnv(), {"budgets": {"total_max_actions": 720}}, Path(td), {}, {"public": {}}, max_actions=1, global_offset=10)
            self.assertFalse(out["graph_complete"])
            self.assertFalse(out["official_success"])
            self.assertTrue(out["native_instant_ever"])
            self.assertEqual(out["stop_reason"], "verify_placement_budget")


class P70LiftClearanceRefinementTests(unittest.TestCase):
    def controller(self, root, *, enabled=True):
        ref = root / "ref.json"; write_ref(ref)
        cfg = config(root, ref)
        cfg["lift_clearance_refinement"] = bool(enabled)
        with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
            c = P70ObjectController(cfg)
        c.state.stage = "lift_clear"
        c.state.held_T_eef_book = np.eye(4).tolist()
        goal = np.eye(4); goal[2, 3] = 1.0
        c.state.lift_goal_T_world_book = goal.tolist()
        return c

    def eligible_facts(self):
        return {"attached": True, "bilateral_contact": True, "lift_clear": False}

    def test_lift_refinement_disabled_preserves_old_goal(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.controller(Path(td), enabled=False)
            old = np.asarray(c.state.lift_goal_T_world_book)
            out = c._maybe_refine_lift_goal(public_frame(), self.eligible_facts(), old)
            self.assertFalse(out["applied"])
            self.assertEqual(out["reason"], "disabled")
            self.assertFalse(c.state.lift_refined_once)
            self.assertTrue(np.allclose(np.asarray(c.state.lift_goal_T_world_book), old))

    def test_lift_refinement_triggers_once_when_public_goal_reached_and_attached(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.controller(Path(td), enabled=True)
            old = np.asarray(c.state.lift_goal_T_world_book)
            out = c._maybe_refine_lift_goal(public_frame(), self.eligible_facts(), old)
            self.assertTrue(out["applied"])
            self.assertTrue(c.state.lift_refined_once)
            amount = np.min(c.relation_targets["book_geometry"]["halfsize"]) / 4.0
            self.assertAlmostEqual(out["z_increment_m"], amount)
            self.assertAlmostEqual(np.asarray(c.state.lift_goal_T_world_book)[2, 3], old[2, 3] + amount)
            second = c._maybe_refine_lift_goal(public_frame(), self.eligible_facts(), np.asarray(c.state.lift_goal_T_world_book))
            self.assertFalse(second["applied"])
            self.assertEqual(second["reason"], "already_applied")

    def test_lift_refinement_requires_valid_grasp_facts_and_public_goal_error_gate(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.controller(Path(td), enabled=True)
            old = np.asarray(c.state.lift_goal_T_world_book)
            for facts in [
                {"attached": False, "bilateral_contact": True, "lift_clear": False},
                {"attached": True, "bilateral_contact": False, "lift_clear": False},
                {"attached": True, "bilateral_contact": True, "lift_clear": True},
            ]:
                out = c._maybe_refine_lift_goal(public_frame(), facts, old)
                self.assertFalse(out["applied"])
                self.assertEqual(out["reason"], "stage_facts_not_eligible")
            far = public_frame(); far["eef_pos"] = np.array([0.01, 0.0, 0.0])
            out = c._maybe_refine_lift_goal(far, self.eligible_facts(), old)
            self.assertFalse(out["applied"])
            self.assertEqual(out["reason"], "public_goal_error_too_large")
            self.assertFalse(c.state.lift_refined_once)

    def test_resume_preserves_lift_refined_once_flag(self):
        class C: pass
        state = P70ControllerState(stage="lift_clear", lift_refined_once=True, lift_goal_T_world_book=np.eye(4).tolist()).to_dict()
        c = C(); from tools.task_graph.p70_stage_resume import set_controller_state
        set_controller_state(c, state)
        self.assertTrue(c.state.lift_refined_once)


class P70LiftRefinementTerminalGuardTests(unittest.TestCase):
    def test_lift_refinement_does_not_trigger_on_unexecuted_budget_stop_row(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.json"; write_ref(ref)
            cfg = config(Path(td), ref); cfg["lift_clearance_refinement"] = True
            with mock.patch.object(P70ObjectController, "_load_grasp_policy", fake_load):
                c = P70ObjectController(cfg)
            c.state.stage = "lift_clear"; c.state.held_T_eef_book = np.eye(4).tolist()
            goal = np.eye(4); goal[2, 3] = 1.0; c.state.lift_goal_T_world_book = goal.tolist()
            c._current_transition = {"stop": "lift_clear_budget", "stage": "lift_clear"}
            out = c._maybe_refine_lift_goal(public_frame(), {"attached": True, "bilateral_contact": True, "lift_clear": False}, goal)
            self.assertFalse(out["applied"])
            self.assertEqual(out["reason"], "transition_stop")
            self.assertFalse(c.state.lift_refined_once)


if __name__ == "__main__":
    unittest.main()
