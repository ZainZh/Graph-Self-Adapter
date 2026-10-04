from dataclasses import replace

import numpy as np
import pytest

from prove.task_graph.fast_adapter_bridge import (
    AdapterProfile, CONTROLLER_FORMAT, FastAdapterStageBridge,
    compatibility_issues, paired_bridges,
)


class Model:
    task = 'insert'
    representation = 'geometry'

    def to(self, device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, value):
        self.trainable = value
        return self


class Learner:
    def __init__(self, model, arm, device):
        self.model, self.arm = model, arm
        self.records, self.z = [], None
        self.last = None

    def update(self, obs, t):
        if obs['corners'] is None:
            return
        self.last = obs
        self.z = 0 if self.z is None else self.z + (self.arm == 'continuous')
        self.records.append({'t': t, 'z': self.z})

    def estimate(self):
        return self.z


def profile():
    return AdapterProfile('insert', 'preheld_peg_slot_insertion', CONTROLLER_FORMAT,
                          (448, 448, 3), True, True, True, True, True)


def observation():
    return dict(corners=np.array([[100, 100], [200, 100], [200, 200], [100, 200]]),
                camera_position=np.zeros(3), camera_rotation=np.eye(3), K=np.eye(3),
                position=np.zeros(3), rotation=np.eye(3), role_bbox=np.zeros((3, 4)),
                role_visible=np.ones(3))


def bridge(**kwargs):
    return FastAdapterStageBridge(Model(), learner_factory=Learner, **kwargs)


def test_latent_persists_only_inside_stage_and_paired_weights_identical():
    model = Model()
    pair = paired_bridges(model, learner_factory=Learner)
    assert pair['fixed'].model is pair['continuous'].model is model
    assert model.trainable is False
    for item in pair.values():
        item.start_stage('first', profile())
        assert item.step(observation(), 0) == 0
    assert pair['fixed'].step(observation(), 1) == 0
    assert pair['continuous'].step(observation(), 1) == 1
    item = pair['continuous']
    assert item.finish_stage()['records'][-1]['z'] == 1
    with pytest.raises(RuntimeError):
        item.step(observation(), 2)
    item.start_stage('second', profile())
    assert item.step(observation(), 0) == 0
    item.stop()
    assert item._learner is None


@pytest.mark.parametrize('task', ['book_to_caddy', 'bowl_to_drawer', 'mug_to_microwave'])
def test_placement_cannot_be_renamed_insert(task):
    incompatible = replace(profile(), interaction=task, rigid_preheld_object=False)
    assert compatibility_issues(incompatible, 'insert')
    with pytest.raises(ValueError, match='outside the frozen task domain'):
        bridge().start_stage(task, incompatible)


@pytest.mark.parametrize('field,value', [
    ('wrist_rgb_shape', (256, 256, 3)), ('calibrated_wrist', False),
    ('four_visible_coplanar_corners', False), ('visible_role_masks', False),
    ('eef_pose', False), ('controller_format', 'libero_7d'), ('task', 'cup'),
])
def test_profile_mismatch_rejected(field, value):
    with pytest.raises(ValueError):
        bridge().start_stage('s', replace(profile(), **{field: value}))


@pytest.mark.parametrize('field', ['success', 'oracle', 'target_pose', 'supervisor', 'evaluator'])
def test_truth_fields_rejected(field):
    item = bridge()
    item.start_stage('s', profile())
    obs = observation()
    obs[field] = True
    with pytest.raises(ValueError, match='only the legacy public fields'):
        item.step(obs, 0)
    assert item._learner.records == []


def test_visibility_lock_and_time_order():
    item = bridge()
    item.start_stage('s', profile())
    obs = observation()
    obs['corners'] = None
    assert item.step(obs, 0) is None
    assert item.step(observation(), 1) == 0
    with pytest.raises(ValueError):
        item.step(observation(), 1)
    with pytest.raises(RuntimeError):
        item.start_stage('another', profile())
    item.freeze_adaptation()
    assert item.step(observation(), 2) == 0
    assert len(item.finish_stage()['records']) == 1


def test_observation_copied_and_bad_features_rejected():
    item = bridge()
    item.start_stage('s', profile())
    obs = observation()
    item.step(obs, 0)
    obs['position'][:] = 100
    assert np.all(item._learner.last['position'] == 0)
    obs['K'][0, 0] = np.nan
    with pytest.raises(ValueError):
        item.step(obs, 1)
    model = Model()
    model.representation = 'relative_fusion'
    item = FastAdapterStageBridge(model, learner_factory=Learner)
    item.start_stage('s', profile())
    with pytest.raises(ValueError, match='visual features'):
        item.step(observation(), 0)


@pytest.mark.parametrize('arm', ['fixed', 'continuous'])
@pytest.mark.parametrize('family', ['p45', 'p47'])
def test_actual_frozen_learner_interface_parity_synthetic_untrained(arm, family):
    """Interface regression only: random weights and synthetic public geometry."""
    torch = pytest.importorskip('torch')
    from prove.fast_adapter.geometry3d_adapter import project
    from prove.fast_adapter.p45_neural import P45NeuralGeometry, P45Latent
    from prove.fast_adapter.p47_neural import P47Latent

    torch.manual_seed(710)
    model = P45NeuralGeometry('insert', 'geometry')
    weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    item = FastAdapterStageBridge(model, arm=arm, family=family)
    raw_cls = P47Latent if family == 'p47' else P45Latent
    raw = raw_cls(model, arm=arm)
    item.start_stage('synthetic', profile())
    corners_world = np.array([[-.019, .016, 0], [.019, .016, 0],
                              [.019, -.016, 0], [-.019, -.016, 0]])
    for t in range(6):
        obs = observation()
        obs['camera_position'] = np.array([.008 * t, 0, .3], dtype=np.float32)
        obs['K'] = np.array([[420, 0, 224], [0, 420, 224], [0, 0, 1]], dtype=np.float32)
        obs['corners'] = project(corners_world, obs).astype(np.float32)
        obs['local'] = np.zeros(48, dtype=np.float32)
        actual = item.step(obs, t)
        raw.update(obs, t)
        expected = raw.estimate()
        assert len(actual) == 3
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    actual_records = item.finish_stage()['records']
    assert actual_records == raw.records
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key])
    assert all(not parameter.requires_grad for parameter in model.parameters())
    item.start_stage('reset', profile())
    assert item._learner.z is None
    assert item._learner.history == []
    assert item._learner.reference is None
