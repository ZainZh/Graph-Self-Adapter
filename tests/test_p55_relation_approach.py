import numpy as np
import pytest

from prove.task_graph.p55_relation_approach import PublicRelationApproach


def references():
    return (dict(position=[0., 0., 1.], quaternion_xyzw=[0., 0., 0., 1.]),
            dict(position=[0., 0., 0.8], quaternion_xyzw=[0., 0., 0., 1.]))


def test_public_axis_continues_with_original_osc_limits():
    controller = PublicRelationApproach(*references())
    action, diagnostic = controller.command([0., 0., .8], [0., 0., 0., 1.])
    np.testing.assert_allclose(diagnostic['direction'], [0., 0., -1.])
    np.testing.assert_allclose(action, [0., 0., -.2, 0., 0., 0., 1.], atol=1e-12)
    assert np.linalg.norm(action[:3]*.05) <= .01+1e-12
    assert np.linalg.norm(action[3:6]*.5) <= .05+1e-12


def test_lateral_correction_uses_only_public_eef_and_reference_line():
    controller = PublicRelationApproach(*references())
    action, diagnostic = controller.command([.006, -.002, .8], [0., 0., 0., 1.])
    np.testing.assert_allclose(diagnostic['lateral_error'], [.006, -.002, 0.], atol=1e-12)
    assert action[0] < 0 and action[1] > 0 and action[2] < 0
    assert np.linalg.norm(action[:3]*.05) <= .01+1e-12
    assert 'scale' not in diagnostic and 'private_geometry' not in diagnostic


def test_degenerate_public_axis_is_rejected():
    a, _ = references()
    with pytest.raises(ValueError):
        PublicRelationApproach(a, a)
