import unittest

import torch

from prove.fast_adapter.geometry_role_dynamics import GEOMETRY_MODES, GeometryRoleDynamics, geometry_features


class P36GeometryTest(unittest.TestCase):
    def test_capacity_matches(self):
        counts = {mode: sum(param.numel() for param in GeometryRoleDynamics(mode).parameters()) for mode in GEOMETRY_MODES}
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_hidden_geometry_is_zero(self):
        boxes = torch.tensor([[[0.2, 0.3, 0.1, 0.2], [0.7, 0.8, 0.2, 0.1]]])
        velocity = torch.ones_like(boxes)
        hidden = torch.tensor([[False, False]])
        actor_only = torch.tensor([[True, False]])
        for mode in GEOMETRY_MODES:
            self.assertEqual(float(geometry_features(boxes, velocity, hidden, mode).abs().max()), 0.0)
            g = geometry_features(boxes, velocity, actor_only, mode)
            self.assertEqual(float(g[..., 16:20].abs().max()), 0.0)
            self.assertEqual(float(g[..., 22:].abs().max()), 0.0)

    def test_explicit_adds_relation_terms(self):
        boxes = torch.tensor([[[0.2, 0.3, 0.1, 0.2], [0.7, 0.8, 0.2, 0.1]]])
        velocity = torch.zeros_like(boxes)
        valid = torch.tensor([[True, True]])
        existing = geometry_features(boxes, velocity, valid, "p34_existing_bbox_geometry")
        explicit = geometry_features(boxes, velocity, valid, "p36_explicit_relation_geometry")
        self.assertEqual(float(existing[..., 22:].abs().max()), 0.0)
        self.assertGreater(float(explicit[..., 22:].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
