"""P28 shared IO and task-level paths; upstream environments kept unmodified."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

R = Path('evaluations/fast_adapter_p28_metaworld_v01')
DEPS = Path('data/runtime/fast-adapter-p28-deps')
TASKS = ('pick-place-v3', 'drawer-open-v3')
STRATA = ('native', 'response_ood')
HORIZON = 160
SUPPORT = 80
BLOCK = 16
SEEDS = (11, 22)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def weight_hash(net):
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def task_root(task):
    assert task in TASKS
    return R/'tasks'/task


def transform(seed, mode):
    if mode == 'native':
        return np.eye(3, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if mode == 'train':
        angle = rng.uniform(-30, 30)
        gains = rng.uniform(.7, 1.3, 3)
    elif mode == 'response_ood':
        angle = rng.choice([-1, 1])*rng.uniform(35, 50)
        gains = rng.uniform(.6, 1.4, 3)
    else:
        raise ValueError(mode)
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return (rotation@np.diag(gains)).astype(np.float32)


class TaskEnv:
    """Only public() and step() outputs may enter a learned runtime policy.

    Teacher/raw state/metrics remain inside the collection/evaluation process.
    Fixed horizon: no reward, success, or stage feedback controls execution.
    """
    def __init__(self, task, matrix=None, render=True):
        sys.path.insert(0, str(DEPS.resolve()))
        import mujoco
        from metaworld.env_dict import ENV_CLS_MAP
        self.env = ENV_CLS_MAP[task]()
        self.env._set_task_called = True
        self.env._freeze_rand_vec = False
        # Match upstream observable-goal construction, including its cache reset.
        self.env._partially_observable = False
        del self.env.sawyer_observation_space
        self.env.seeded_rand_vec = True
        self.task = task
        self.matrix = np.eye(3) if matrix is None else np.array(matrix, copy=True)
        self.renderer = mujoco.Renderer(self.env.model, height=224, width=224) if render else None
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0, .7, .12]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.05, 145, -35
        self.obs = None
        self.metrics = []

    def reset(self, seed):
        self.env.seed(int(seed))
        self.obs, _ = self.env.reset()
        self.metrics = []
        return self.public()

    def public(self):
        return self.obs[:4].astype(np.float32).copy()

    def rgb(self):
        self.renderer.update_scene(self.env.data, camera=self.cam)
        return self.renderer.render().copy()

    def step(self, action):
        # The adapter sees its requested action, never this hidden transform.
        applied = np.asarray(action, np.float64).copy()
        applied[:3] = self.matrix@applied[:3]
        applied = np.clip(applied, -1, 1)
        self.obs, _, _, _, info = self.env.step(applied)
        self.metrics.append({k: float(v) for k, v in info.items()})
        return self.public()

    def teacher(self):
        from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy
        from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
        p = SawyerPickPlaceV3Policy() if self.task == TASKS[0] else SawyerDrawerOpenV3Policy()
        nominal = np.clip(p.get_action(self.obs), -1, 1)
        nominal[:3] = np.linalg.solve(self.matrix, nominal[:3])
        return np.clip(nominal, -1, 1).astype(np.float32)

    def state(self):
        return {k: getattr(self.env.data, k).copy() for k in
                ('qpos', 'qvel', 'mocap_pos', 'mocap_quat')}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.env.close()


class Visual:
    def __init__(self, dino, pca=None):
        from tools.fast_adapter.p22_data import RGBEncoder
        self.net = RGBEncoder(dino)
        self.pca = None if pca is None else {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in pca.items()}

    def encode(self, images):
        f = self.net.encode(images)
        if self.pca is None:
            return f
        return (f-self.pca['mean'])@self.pca['basis']/self.pca['sd']


def load_base(task):
    from prove.fast_adapter.metaworld_adapter import RGBSkill
    model = RGBSkill().cuda()
    blob = torch.load(task_root(task)/'models/base_final.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(blob['state'])
    return model.eval().requires_grad_(False)


def load_adapter(task, seed, variant):
    from prove.fast_adapter.metaworld_adapter import RobotAdapter
    blob = torch.load(task_root(task)/'models'/f'seed{seed}'/f'{variant}.pt',
                      map_location='cpu', weights_only=True)
    net = RobotAdapter(history=variant == 'history').cuda()
    net.load_state_dict(blob['state'])
    return net.eval().requires_grad_(False), blob
