"""Train-only ridge measurement and runtime ray geometry for P58 hammer."""
from __future__ import annotations

import numpy as np

from prove.task_graph.p57_public_geometry import rays


REPRESENTATIONS = ("absolute_positions", "role_relative")
PCA_DIMS = (8, 16, 32)
RIDGE = (.01, .1, 1., 10.)


def public_features(frame):
    """Flatten only the P58 public allowlist, excluding RGB/mask pixels."""
    token = np.asarray(frame["mask_tokens"], np.float64).reshape(-1)
    valid = np.asarray(frame["mask_valid"], np.float64).reshape(-1)
    K = np.asarray(frame["camera_K"], np.float64)
    camera = np.asarray(frame["camera_pose"], np.float64)
    calib = np.stack((K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]), -1).reshape(-1)
    camera_public = np.concatenate((camera[:, :3, 3].reshape(-1),
                                    camera[:, :3, :2].reshape(-1)))
    eef = np.asarray(frame["eef_transform"], np.float64)
    eef_public = np.concatenate((eef[:3, 3], eef[:3, :2].reshape(-1)))
    proprio = np.asarray(frame["proprio"], np.float64).reshape(-1)
    result = np.concatenate((token, valid, calib, camera_public, eef_public, proprio))
    if result.shape != (263,) or not np.isfinite(result).all():
        raise ValueError("P58 hammer public feature contract changed")
    return result.astype(np.float32)


def relative_uv(frame, semantic_uv):
    """Semantic centers relative to each visible role mask box."""
    tokens = np.asarray(frame["mask_tokens"], np.float64)
    valid = np.asarray(frame["mask_valid"], bool)
    boxes = tokens[..., :4] * 448.
    center = (boxes[..., :2] + boxes[..., 2:]) * .5
    extent = (boxes[..., 2:] - boxes[..., :2]).clip(1.)
    target = (np.asarray(semantic_uv, np.float64) - center) / extent
    return target.reshape(-1).astype(np.float32), np.repeat(valid[..., None], 2, -1).reshape(-1)


def fit_pca(x, dim):
    mean = x.mean(0); scale = x.std(0).clip(1e-6)
    normalized = (x - mean) / scale
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    use = min(int(dim), len(vt))
    components = vt[:use].T
    scores = normalized @ components
    score_scale = scores.std(0).clip(1e-6)
    return dict(mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                components=components.astype(np.float32), score_scale=score_scale.astype(np.float32))


def transform_pca(model, x):
    return ((x-model["mean"])/model["scale"] @ model["components"])/model["score_scale"]


def fit_ridge(x, y, valid, strength):
    design = np.c_[x, np.ones(len(x))]
    coef = np.zeros((design.shape[1], y.shape[1]), np.float64)
    eye = np.eye(design.shape[1]); eye[-1, -1] = 0.
    for column in range(y.shape[1]):
        keep = valid[:, column]
        coef[:, column] = np.linalg.solve(design[keep].T@design[keep] + strength*eye,
                                          design[keep].T@y[keep, column])
    return coef.astype(np.float32)


def predict_relative(model, features):
    score = transform_pca(model["pca"], np.asarray(features, np.float64))
    return np.c_[score, np.ones(len(score))] @ model["coef"]


def load_package(path, expected_sha256=None):
    """Load a frozen common-UV plus representation-specific XYZ ridge package."""
    if expected_sha256 is not None:
        import hashlib
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("P58 hammer readout package hash changed")
    blob = np.load(path, allow_pickle=False)
    required = {f"{head}_{key}" for head in ("uv", "xyz")
                for key in ("mean", "scale", "components", "score_scale", "coef")}
    if set(blob.files) != required:
        raise ValueError("P58 hammer readout package keys changed")
    def head(name):
        return dict(pca=dict(mean=blob[f"{name}_mean"], scale=blob[f"{name}_scale"],
            components=blob[f"{name}_components"], score_scale=blob[f"{name}_score_scale"]),
            coef=blob[f"{name}_coef"])
    return dict(uv=head("uv"), xyz=head("xyz"))


def predict_measurement(package, frame):
    feature = public_features(frame)[None]
    uv = pixels_from_relative(predict_relative(package["uv"], feature)[0], frame)
    xyz = predict_relative(package["xyz"], feature)[0].reshape(2, 3).astype(np.float32)
    return uv.astype(np.float32), xyz


def pixels_from_relative(relative, frame):
    rel = np.asarray(relative).reshape(2, 2, 2)
    boxes = np.asarray(frame["mask_tokens"])[..., :4] * 448.
    center = (boxes[..., :2] + boxes[..., 2:]) * .5
    extent = (boxes[..., 2:] - boxes[..., :2]).clip(1.)
    return center + extent * rel


def ray_system(uv, frame, representation):
    """Return per-role ray A/C in the representation's declared coordinates."""
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown P58 hammer representation")
    uv = np.asarray(uv, np.float64)
    valid = np.asarray(frame["mask_valid"], bool).T.copy()
    K = np.asarray(frame["camera_K"], np.float64)
    camera = np.asarray(frame["camera_pose"], np.float64)
    eef = np.asarray(frame["eef_transform"], np.float64)
    A = np.zeros((2, 2, 3, 3), np.float32); C = np.zeros((2, 2, 3), np.float32)
    for view in range(2):
        for role in range(2):
            origin, direction = rays(uv[view, role], K[view], camera[view])
            if representation == "role_relative" and role == 0:
                origin = eef[:3, :3].T @ (origin-eef[:3, 3])
                direction = eef[:3, :3].T @ direction
            A[role, view] = np.eye(3)-np.outer(direction, direction)
            C[role, view] = origin
    return A, C, valid


def solve_points(A, C, valid):
    points = np.zeros((2, 3), np.float32); observable = np.zeros(2, bool)
    for role in range(2):
        keep = np.asarray(valid[role], bool)
        if keep.sum() < 2:
            continue
        matrix = A[role, keep].sum(0)
        if np.linalg.matrix_rank(matrix, tol=1e-5) < 3:
            continue
        points[role] = np.linalg.solve(matrix,
            np.einsum("vij,vj->i", A[role, keep], C[role, keep]))
        observable[role] = True
    return points, observable


def to_absolute(points, eef_transform, representation):
    points = np.asarray(points, np.float64).copy()
    if representation == "role_relative":
        eef = np.asarray(eef_transform, np.float64)
        points[0] = eef[:3, :3]@points[0] + eef[:3, 3]
    return points.astype(np.float32)
