"""Public-observation isolation and absent-region checks for P43."""
import ast
import inspect
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter import neural_geometry_visual as visual
from prove.fast_adapter.geometry3d_env import GeometrySkill, configuration
from tools.fast_adapter.p43_data import ROOT, js


def forbidden(*args, **kwargs):
    raise AssertionError('Privileged field was read by public observation')


class Poison:
    __getitem__ = forbidden
    __iter__ = forbidden


def main():
    records = []
    for task in ('cup', 'insert'):
        env = GeometrySkill(configuration(task, 439000))
        try:
            env.cfg = Poison(); env.truth_target = Poison()
            env.oracle = forbidden; env.facts = forbidden
            obs, rgb, masks = visual.observe_roles(env)
            env.goal_ids = []
            absent, _, missing_masks = visual.observe_roles(env)
            assert absent['corners'] is None
            assert not absent['role_visible'][2]
            assert not absent['role_bbox'][2].any()
            assert not missing_masks[2].any()
            records.append(dict(task=task, public_observer_works_with_privileged_fields_poisoned=True,
                                absent_goal_corners_none=True, absent_goal_bbox_zero=True,
                                absent_goal_mask_zero=True))
        finally:
            env.close()
    network = visual.FrozenRoleVision.__new__(visual.FrozenRoleVision)
    network.device = torch.device('cpu')
    network.set_pca(dict(fit_split='train', mean=torch.ones(3, 3072),
                        basis=torch.ones(3, 3072, 16), scale=torch.ones(3, 16)))
    projected = network.project(torch.zeros(2, 3, 3072), torch.zeros(2, 3, dtype=torch.bool))
    assert projected.shape == (2, 48)
    assert projected.abs().max() == 0
    rejected = False
    try:
        network.set_pca(dict(fit_split='dev'))
    except ValueError:
        rejected = True
    assert rejected
    tree = ast.parse(inspect.getsource(visual))
    forbidden_names = {'cfg', 'oracle', 'facts', 'truth_target', 'geom_size', 'xpos', 'xmat'}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (attrs & forbidden_names)
    result = dict(passed=True, observations=records, absent_mask_after_pca_exact_zero=True,
                  nontraining_pca_rejected=True, forbidden_runtime_attributes=sorted(forbidden_names),
                  forbidden_attribute_accesses=sorted(attrs & forbidden_names))
    js(ROOT/'visual/ISOLATION_CHECKS.json', result)
    print(result)


if __name__ == '__main__':
    main()
