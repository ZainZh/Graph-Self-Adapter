import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from prove.task_graph.p70_object_controller import P70ControllerState
from tools.task_graph import p70_stage_resume as resume


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeSim:
    def __init__(self):
        self.state = np.zeros(45, dtype=np.float64)
        self.forward_called = False
        self.reset_called = False
    def reset(self): self.reset_called = True
    def set_state_from_flattened(self, value): self.state = np.asarray(value, dtype=np.float64).copy()
    def get_state(self):
        class S:
            def __init__(self, v): self.v = v
            def flatten(self): return self.v.copy()
        return S(self.state)
    def forward(self): self.forward_called = True


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim(); self.post = False; self.updated = False
    def _post_process(self): self.post = True
    def _update_observables(self, force=False): self.updated = bool(force)
    def _get_observations(self, force_update=False): return {"force_update": bool(force_update)}


def write_snapshot(path: Path, stage="lift_clear", include_ctrl=True, state_offset=0.0):
    state = np.arange(45, dtype=np.float64) + float(state_offset)
    controller = P70ControllerState(stage=stage, t=7, stage_t=2, held_T_eef_book=np.eye(4).tolist()).to_dict()
    payload = dict(sim_state_flattened=state, controller_state=np.asarray(json.dumps(controller), dtype=object))
    if include_ctrl:
        payload["supervisor_ctrl"] = np.zeros(2, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return state, controller


def write_fake_cell(root: Path, *, seal_snapshot=True):
    cell = root / "cell_000"; (cell / "prefix" / "private").mkdir(parents=True)
    snap = cell / "prefix" / "private" / "stage_entry_001_lift_clear.npz"
    write_snapshot(snap, "lift_clear")
    q0_freeze = cell / "BINDING.json"; q0_freeze.write_text(json.dumps({"schema": "q0"}))
    files = {"BINDING.json": sha(q0_freeze)}
    if seal_snapshot:
        files["prefix/private/stage_entry_001_lift_clear.npz"] = sha(snap)
    seal = cell / "SEAL.json"; seal.write_text(json.dumps({"schema": "seal", "files": files}))
    result = cell / "RESULT.json"; result.write_text(json.dumps({"schema": resume.Q0_RESULT_SCHEMA, "stage_qualification_only": True, "seal_sha256": sha(seal)}))
    return cell, result, seal, q0_freeze, snap


class StageResumeTests(unittest.TestCase):
    def test_snapshot_roundtrip_and_restore_exact_45d(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "snap.npz"; state, _ = write_snapshot(path)
            loaded_state, loaded_controller = resume.load_snapshot(path)
            self.assertTrue(np.array_equal(loaded_state, state))
            self.assertEqual(loaded_controller["stage"], "lift_clear")
            env = FakeEnv(); obs = resume.restore_snapshot(env, loaded_state)
            self.assertTrue(env.sim.reset_called)
            self.assertTrue(env.sim.forward_called)
            self.assertTrue(env.post)
            self.assertTrue(env.updated)
            self.assertEqual(obs["force_update"], True)
            self.assertTrue(np.array_equal(env.sim.state, state))

    def test_set_controller_state_preserves_saved_state_and_stage_override(self):
        class C: pass
        c = C(); state = P70ControllerState(stage="lift_clear", t=5, stage_t=4, held_T_eef_book=np.eye(4).tolist()).to_dict()
        resume.set_controller_state(c, state, stage="orient_clear")
        self.assertEqual(c.state.stage, "orient_clear")
        self.assertEqual(c.state.t, 5)
        self.assertEqual(c.state.stage_t, 4)
        self.assertIsNotNone(c.state.held_T_eef_book)

    def test_closed_hold_action_keeps_gripper_closed(self):
        action = resume.closed_hold_action()
        self.assertEqual(action.shape, (7,))
        self.assertTrue(np.allclose(action[:6], 0.0))
        self.assertEqual(float(action[6]), 1.0)

    def test_trace_record_renames_controller_t_and_outer_fields_win(self):
        row = resume.trace_record(11, {"t": 4, "facts": {"inner": True}, "phase": "inner"}, facts={"outer": True}, phase="outer")
        self.assertEqual(row["t"], 11)
        self.assertEqual(row["controller_t"], 4)
        self.assertEqual(row["facts"], {"outer": True})
        self.assertEqual(row["phase"], "outer")

    def test_prepare_rejects_unsealed_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, result, _, _, snap = write_fake_cell(root, seal_snapshot=False)
            with mock.patch.object(resume, "validate_q0_freeze", return_value={"controller": {}, "p55_binding": {}}), \
                 mock.patch.object(resume, "source_bindings", return_value={}):
                with self.assertRaisesRegex(ValueError, "not sealed"):
                    resume.prepare(output=root / "out", q0_result=result, stage_snapshot=snap, stage="lift_clear", budget=10)

    def test_prepare_accepts_sealed_snapshot_and_binds_source_q0(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, result, _, _, snap = write_fake_cell(root, seal_snapshot=True)
            q0 = {"controller": {"ok": True}, "p55_binding": {"path": "p55", "sha256": "x"}, "controller_file_bindings": {}}
            with mock.patch.object(resume, "validate_q0_freeze", return_value=q0), \
                 mock.patch.object(resume, "source_bindings", return_value={}):
                frozen = resume.prepare(output=root / "out", q0_result=result, stage_snapshot=snap, stage="lift_clear", budget=10)
            self.assertEqual(frozen["candidate_controller_source"], "source_q0_controller")
            self.assertEqual(frozen["candidate_controller"], {"ok": True})

    def test_prepare_rejects_actor_and_insert_stages_and_bad_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, result, _, _, snap = write_fake_cell(root, seal_snapshot=True)
            with self.assertRaisesRegex(ValueError, "held stages"):
                resume.prepare(output=root / "a", q0_result=result, stage_snapshot=snap, stage="short_grasp", budget=10)
            with self.assertRaisesRegex(ValueError, "1..240"):
                resume.prepare(output=root / "b", q0_result=result, stage_snapshot=snap, stage="lift_clear", budget=241)

    def test_prepare_suffix_accepts_orient_to_near_entry_budget392(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cell, result, seal, q0_freeze, old_snap = write_fake_cell(root, seal_snapshot=False)
            snap = cell / "prefix" / "private" / "stage_entry_002_orient_clear.npz"
            write_snapshot(snap, "orient_clear")
            data = json.loads(seal.read_text()); data["files"]["prefix/private/stage_entry_002_orient_clear.npz"] = sha(snap); seal.write_text(json.dumps(data))
            result.write_text(json.dumps({"schema": resume.Q0_RESULT_SCHEMA, "stage_qualification_only": True, "seal_sha256": sha(seal)}))
            q0 = {"controller": {"ok": True}, "p55_binding": {"path": "p55", "sha256": "x"}, "controller_file_bindings": {}}
            with mock.patch.object(resume, "validate_q0_freeze", return_value=q0), \
                 mock.patch.object(resume, "source_bindings", return_value={}):
                frozen = resume.prepare(output=root / "out", q0_result=result, stage_snapshot=snap, stage="orient_clear", budget=392, mode="suffix", stop_after="near_entry")
            self.assertEqual(frozen["mode"], "suffix")
            self.assertEqual(frozen["stop_after"], "near_entry")
            self.assertTrue(frozen["runtime_contract"]["p68_not_executed"])

    def test_prepare_suffix_rejects_wrong_start_or_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, result, _, _, snap = write_fake_cell(root, seal_snapshot=True)
            with self.assertRaisesRegex(ValueError, "orient_clear"):
                resume.prepare(output=root / "a", q0_result=result, stage_snapshot=snap, stage="lift_clear", budget=392, mode="suffix", stop_after="near_entry")
            orient = root / "cell_000" / "prefix" / "private" / "stage_entry_002_orient_clear.npz"
            write_snapshot(orient, "orient_clear")
            seal = root / "cell_000" / "SEAL.json"; data = json.loads(seal.read_text()); data["files"]["prefix/private/stage_entry_002_orient_clear.npz"] = sha(orient); seal.write_text(json.dumps(data))
            result.write_text(json.dumps({"schema": resume.Q0_RESULT_SCHEMA, "stage_qualification_only": True, "seal_sha256": sha(seal)}))
            with self.assertRaisesRegex(ValueError, "1..392"):
                resume.prepare(output=root / "b", q0_result=result, stage_snapshot=orient, stage="orient_clear", budget=393, mode="suffix", stop_after="near_entry")

    def test_prepare_accepts_paired_control_snapshot_when_stage_has_no_ctrl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cell, result, seal, q0_freeze, snap = write_fake_cell(root, seal_snapshot=False)
            write_snapshot(snap, "lift_clear", include_ctrl=False)
            control = cell / "prefix" / "private" / "frame_0220.npz"
            write_snapshot(control, "lift_clear", include_ctrl=True)
            data = json.loads(seal.read_text())
            data["files"]["prefix/private/stage_entry_001_lift_clear.npz"] = sha(snap)
            data["files"]["prefix/private/frame_0220.npz"] = sha(control)
            seal.write_text(json.dumps(data))
            result.write_text(json.dumps({"schema": resume.Q0_RESULT_SCHEMA, "stage_qualification_only": True, "seal_sha256": sha(seal)}))
            q0 = {"controller": {"ok": True}, "p55_binding": {"path": "p55", "sha256": "x"}, "controller_file_bindings": {}}
            with mock.patch.object(resume, "validate_q0_freeze", return_value=q0), \
                 mock.patch.object(resume, "source_bindings", return_value={}):
                frozen = resume.prepare(output=root / "out_pair", q0_result=result, stage_snapshot=snap, control_snapshot=control, stage="lift_clear", budget=10)
            self.assertEqual(Path(frozen["source"]["control_snapshot"]["path"]).name, "frame_0220.npz")

    def test_prepare_rejects_unpaired_control_snapshot_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cell, result, seal, q0_freeze, snap = write_fake_cell(root, seal_snapshot=False)
            write_snapshot(snap, "lift_clear", include_ctrl=False, state_offset=0.0)
            control = cell / "prefix" / "private" / "frame_bad.npz"
            write_snapshot(control, "lift_clear", include_ctrl=True, state_offset=1.0)
            data = json.loads(seal.read_text())
            data["files"]["prefix/private/stage_entry_001_lift_clear.npz"] = sha(snap)
            data["files"]["prefix/private/frame_bad.npz"] = sha(control)
            seal.write_text(json.dumps(data))
            result.write_text(json.dumps({"schema": resume.Q0_RESULT_SCHEMA, "stage_qualification_only": True, "seal_sha256": sha(seal)}))
            q0 = {"controller": {"ok": True}, "p55_binding": {"path": "p55", "sha256": "x"}, "controller_file_bindings": {}}
            with mock.patch.object(resume, "validate_q0_freeze", return_value=q0), \
                 mock.patch.object(resume, "source_bindings", return_value={}):
                with self.assertRaisesRegex(ValueError, "does not exactly match"):
                    resume.prepare(output=root / "out_bad", q0_result=result, stage_snapshot=snap, control_snapshot=control, stage="lift_clear", budget=10)


class FakeModel:
    def __init__(self):
        self.actuator_ctrlrange = np.array([[-1., 1.], [0., 2.]], dtype=float)
    def actuator_name2id(self, name):
        return {"left": 0, "right": 1}[name]

class FakeController:
    def __init__(self): self.updated = False; self.reset = False
    def update(self, force=False): self.updated = bool(force)
    def reset_goal(self): self.reset = True

class FakeRobot:
    def __init__(self):
        self.gripper = type("G", (), {"actuators": ["left", "right"], "current_action": None})()
        self.controller = FakeController()

class FakeRestoreEnv:
    def __init__(self):
        self.sim = type("S", (), {})()
        self.sim.data = type("D", (), {"ctrl": np.zeros(2)})()
        self.sim.model = FakeModel()
        self.robots = [FakeRobot()]

class StageResumeControlRestoreTests(unittest.TestCase):
    def test_restore_non45d_control_state_restores_ctrl_gripper_and_goal(self):
        env = FakeRestoreEnv()
        audit = resume.restore_non45d_control_state(env, {"supervisor_ctrl": np.array([0.5, 1.5])})
        self.assertTrue(np.allclose(env.sim.data.ctrl, [0.5, 1.5]))
        self.assertTrue(np.allclose(env.robots[0].gripper.current_action, [0.5, 0.5]))
        self.assertTrue(env.robots[0].controller.updated)
        self.assertTrue(env.robots[0].controller.reset)
        self.assertFalse(audit["object_gt_used_for_action"])

    def test_restore_non45d_control_state_rejects_missing_or_wrong_shape_ctrl(self):
        env = FakeRestoreEnv()
        with self.assertRaisesRegex(ValueError, "supervisor_ctrl"):
            resume.restore_non45d_control_state(env, {})
        with self.assertRaisesRegex(ValueError, "ctrl shape"):
            resume.restore_non45d_control_state(env, {"supervisor_ctrl": np.zeros(3)})


if __name__ == "__main__":
    unittest.main()
