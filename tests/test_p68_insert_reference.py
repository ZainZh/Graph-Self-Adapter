import json
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np
import torch

from prove.fast_adapter.p62_adaptation import candidate_configuration
from prove.fast_adapter.p68_insert_reference_runtime import (
    CONTINUOUS,
    ENTRY_ONCE,
    FRAME_FIXED,
    PREFIX_STEPS,
    REFERENCE_FIXED,
    P68InsertReferenceRuntime,
    p68_candidate_configuration,
)
from tools.task_graph.p68_insert_reference_validation import cells_from_scope, cell_name, load_scope

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "evaluations/fast_adapter_p68_cross_task_reference_v01/INSERT_SCOPE.json"


def public_obs():
    return {
        "K": np.eye(3, dtype=np.float32),
        "camera_position": np.zeros(3, dtype=np.float32),
        "camera_rotation": np.eye(3, dtype=np.float32),
        "position": np.zeros(3, dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    }


def masks(goal=True):
    value = np.zeros((3, 448, 448), dtype=np.uint8)
    if goal:
        value[2, 100:160, 100:160] = 1
    return value


def runtime(arm):
    rt = P68InsertReferenceRuntime.__new__(P68InsertReferenceRuntime)
    rt.device = "cpu"
    rt.arm = arm
    rt.anchor = torch.eye(4)
    rt.z = torch.zeros(8)
    rt.history = []
    rt.records = []
    rt.geometry = None
    rt.reference_geometry = None
    rt.reference_t = None
    rt.entry_update_done = False
    rt.update_events = 0
    rt._estimate = None
    rt.last_t = -1

    def encode(self, public, rgb, mask_value, valid):
        self.anchor = torch.eye(4)
        return torch.tensor([float(self.last_t + 1), 0.0, -1.0, 0.0, 0.0, 0.0, -4.0, -4.1])

    rt._encode_geometry = types.MethodType(encode, rt)
    return rt


class RuntimePatches:
    def __init__(self, admitted=True, support_len=3):
        self.admitted = admitted
        self.support_len = support_len
        self.stack = None

    def __enter__(self):
        import prove.fast_adapter.p68_insert_reference_runtime as mod

        def boundary(mask_value):
            if not self.admitted:
                return None, {}
            hull = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
            return hull, {"hull_vertices": hull}

        def adapt(geometry, z, anchor, support, *, candidate, create_graph):
            out = z.clone()
            out[0] += 0.25
            out[6] += 0.1
            return out, [{"loss": 1.0}, {"loss": 0.5}, {"loss": 0.25}]

        self.stack = mock.patch.multiple(
            mod,
            outer_hull_boundary=boundary,
            causal_support=lambda history: list(history)[-self.support_len:],
            contour_residual=lambda geometry, z, anchor, support: torch.ones(4) * (1.0 + z[0]),
            adapt_z=adapt,
            corrected_components=lambda geometry, z: (geometry[:3] + z[:3], torch.eye(3), torch.exp(geometry[6:8] + z[6:8])),
        )
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stack.__exit__(exc_type, exc, tb)


class P68InsertReferenceTests(unittest.TestCase):
    def test_scope_materialized_48_cells_and_matched_family_ranges(self):
        scope = load_scope(SCOPE)
        cells = cells_from_scope(scope)
        self.assertEqual(len(cells), 48)
        self.assertEqual(cells[0]["arm"], "frame_fixed")
        self.assertEqual(cells[3]["arm"], "continuous")
        self.assertTrue(cell_name(0, cells[0]).startswith("00_id_680100_frame_fixed"))
        self.assertEqual(scope["resource_assignment"]["gpu2"], list(range(24)))
        self.assertEqual(scope["resource_assignment"]["gpu3"], list(range(24, 48)))
        by_family = {}
        for row in scope["conditions"]:
            by_family.setdefault(row["family"], {})[row["split"]] = row["evaluator_configuration"]
        self.assertEqual(set(by_family), {0, 1, 2, 3})
        for rows in by_family.values():
            base = rows["id"]
            for split in ("size", "position"):
                self.assertEqual(rows[split]["angles"], base["angles"])
                self.assertEqual(rows[split]["clearance"], base["clearance"])
            self.assertTrue(np.allclose(np.array(rows["size"]["half"]), np.array(base["half"]) * 1.45))
            self.assertTrue(np.isclose(rows["size"]["length"], base["length"] * 1.45))
            changed = np.not_equal(np.round(rows["position"]["center"][:2], 12), np.round(base["center"][:2], 12))
            self.assertEqual(int(changed.sum()), 1)
            self.assertEqual(rows["position"]["center"][2], base["center"][2])

    def test_update_math_contract_is_p62_translation_size5(self):
        self.assertEqual(p68_candidate_configuration(), candidate_configuration("translation_size5"))
        self.assertEqual(p68_candidate_configuration()["active_indices"], [0, 1, 2, 6, 7])

    def test_reference_lock_requires_admitted_boundary(self):
        rt = runtime(REFERENCE_FIXED)
        with RuntimePatches(admitted=False):
            rt.update(public_obs(), np.zeros((448, 448, 3), dtype=np.uint8), masks(goal=True), 0)
        self.assertIsNone(rt.reference_geometry)
        self.assertFalse(rt.records[-1]["reference_locked_now"])
        with RuntimePatches(admitted=True):
            rt.update(public_obs(), np.zeros((448, 448, 3), dtype=np.uint8), masks(goal=True), 1)
        self.assertEqual(rt.reference_t, 1)
        self.assertTrue(rt.records[-1]["reference_locked_now"])

    def test_reference_waits_after_prefix_instead_of_falling_back_to_current(self):
        rt = runtime(REFERENCE_FIXED)
        with RuntimePatches(admitted=False):
            for t in range(PREFIX_STEPS + 1):
                rt.update(public_obs(), np.zeros((448, 448, 3), dtype=np.uint8), masks(goal=True), t)
        self.assertEqual(rt.records[-1]["estimate_source"], "waiting_reference")
        self.assertIsNone(rt.estimate())

    def test_reference_estimate_persists_when_current_goal_mask_is_missing(self):
        rt = runtime(REFERENCE_FIXED)
        with RuntimePatches(admitted=True):
            rt.update(public_obs(), np.zeros((448, 448, 3), dtype=np.uint8), masks(goal=True), 0)
        with RuntimePatches(admitted=False):
            for t in range(1, PREFIX_STEPS + 1):
                rt.update(public_obs(), np.zeros((448, 448, 3), dtype=np.uint8), masks(goal=False), t)
        self.assertEqual(rt.records[-1]["estimate_source"], "reference")
        self.assertIsNotNone(rt.estimate())

    def test_entry_once_and_continuous_share_first_entry_then_diverge(self):
        c = runtime(ENTRY_ONCE)
        d = runtime(CONTINUOUS)
        with RuntimePatches(admitted=True):
            for rt in (c, d):
                for t in range(PREFIX_STEPS + 2):
                    rt.update(public_obs(), np.zeros((448, 448, 3), dtype=np.uint8), masks(goal=True), t)
        self.assertEqual(c.update_events, 1)
        self.assertEqual([r["update_phase"] for r in c.records if r["U"]], ["entry"])
        self.assertEqual(d.update_events, 2)
        self.assertEqual([r["update_phase"] for r in d.records if r["U"]], ["entry", "continuous"])
        self.assertTrue(torch.equal(c.z[3:6], torch.zeros(3)))
        self.assertTrue(torch.equal(d.z[3:6], torch.zeros(3)))

    def test_frame_fixed_keeps_original_valid_goal_rule(self):
        rt = runtime(FRAME_FIXED)
        with RuntimePatches(admitted=False):
            rt.update(public_obs(), np.zeros((448, 448, 3), dtype=np.uint8), masks(goal=False), 0)
        self.assertEqual(rt.records[-1]["estimate_source"], "current_frame")
        self.assertIsNone(rt.estimate())


if __name__ == "__main__":
    unittest.main()
