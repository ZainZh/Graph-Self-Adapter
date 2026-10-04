"""P57 A2 public-ray translation fit for two predicted semantic origins.

Only public UV predictions, visibility, calibration and EEF poses enter here.
The book history is carried through its EEF frame; the goal is world fixed.
No virtual axis endpoints or private poses are used to infer orientation.
"""
import numpy as np

from prove.task_graph.p57_public_geometry import rays
from prove.task_graph.p57_ray_geometry import _check_pose, _fit

ROLES = ("book", "goal")


def _inputs(pred_uv, role_valid, camera_K, camera_pose, eef_transform,
            history, cond_limit):
    uv = np.asarray(pred_uv, dtype=np.float64)
    valid = np.asarray(role_valid)
    K = np.asarray(camera_K, dtype=np.float64)
    camera = np.asarray(camera_pose, dtype=np.float64)
    eef = np.asarray(eef_transform, dtype=np.float64)
    if uv.ndim != 4 or uv.shape[1:] != (2, 2, 2) or len(uv) < 1:
        raise ValueError("pred_uv must have shape [T,2,2,2]")
    T = len(uv)
    if valid.shape != (T, 2, 2) or valid.dtype != np.bool_:
        raise ValueError("role_valid must be boolean [T,2,2]")
    if K.shape != (T, 2, 3, 3) or camera.shape != (T, 2, 4, 4) or eef.shape != (T, 4, 4):
        raise ValueError("Invalid public K/camera/EEF shapes")
    if not np.isfinite(K).all() or np.any(np.linalg.det(K) == 0):
        raise ValueError("Invalid camera intrinsics")
    _check_pose(camera, "camera")
    _check_pose(eef, "EEF")
    if not np.isfinite(uv[valid]).all():
        raise ValueError("Nonfinite prediction for visible role")
    if isinstance(history, bool) or not isinstance(history, int) or not 1 <= history <= 8:
        raise ValueError("history must be integer in 1..8")
    if not np.isfinite(cond_limit) or cond_limit <= 1:
        raise ValueError("cond_limit must be finite and >1")
    return uv, valid, K, camera, eef


def _window(t, first, uv, valid, K, camera, eef, cond_limit):
    result = {"valid": {}, "rank": {}, "condition": {}, "rms_ray_m": {},
              "reason": {}, "rays": {}, "first_frame": first, "last_frame": t}
    for role_index, role in enumerate(ROLES):
        origins, directions = [], []
        for s in range(first, t + 1):
            for view in range(2):
                if not valid[s, view, role_index]:
                    continue
                origin, direction = rays(uv[s, view, role_index], K[s, view], camera[s, view])
                if role == "book":
                    # In an unchanged grasp, the book origin is fixed in EEF.
                    Rs, Rt = eef[s, :3, :3], eef[t, :3, :3]
                    origin = Rt @ Rs.T @ (origin - eef[s, :3, 3]) + eef[t, :3, 3]
                    direction = Rt @ Rs.T @ direction
                origins.append(origin)
                directions.append(direction)
        point, diagnostic = _fit(np.asarray(origins).reshape(-1, 3),
                                 np.asarray(directions).reshape(-1, 3), cond_limit)
        result[f"p_{role}_world"] = point
        result["valid"][role] = point is not None
        for key in ("rank", "condition", "rms_ray_m", "reason", "rays"):
            result[key][role] = diagnostic[key]
    return result


def fit_origin_translations(pred_uv, role_valid, camera_K, camera_pose,
                            eef_transform, *, history=8, cond_limit=1e5):
    """Return instant/window8 world origins per frame with explicit rejection.

    Shape is [frame]["instant" or "window8"]. Each branch has
    p_book_world and p_goal_world, either a finite length-three vector or None.
    The window is causal and capped at eight frames; it does not fill a missing
    entry from labels or from future observations.
    """
    uv, valid, K, camera, eef = _inputs(
        pred_uv, role_valid, camera_K, camera_pose, eef_transform,
        history, cond_limit)
    return [dict(instant=_window(t, t, uv, valid, K, camera, eef, cond_limit),
                 window8=_window(t, max(0, t-history+1), uv, valid,
                                 K, camera, eef, cond_limit))
            for t in range(len(uv))]


def frames_with_orientation_prior(fit, eef_transform,
                                  mean_book_in_eef_rotation,
                                  mean_goal_world_rotation):
    """Combine public fitted origins with fixed train-only rotations.

    The caller must provide the checkpoint's train-only rotation priors.
    Invalid origin fits remain invalid; orientation does not repair coverage.
    """
    eef = np.asarray(eef_transform, dtype=np.float64)
    held = np.asarray(mean_book_in_eef_rotation, dtype=np.float64)
    goal = np.asarray(mean_goal_world_rotation, dtype=np.float64)
    if eef.shape != (4, 4) or held.shape != (3, 3) or goal.shape != (3, 3):
        raise ValueError("Invalid EEF or orientation prior shapes")
    _check_pose(eef[None], "EEF")
    for name, R in (("book prior", held), ("goal prior", goal)):
        pose = np.eye(4)
        pose[:3, :3] = R
        _check_pose(pose[None], name)
    if not fit["valid"]["book"] or not fit["valid"]["goal"]:
        return None
    book = np.eye(4)
    desired = np.eye(4)
    book[:3, :3] = eef[:3, :3] @ held
    desired[:3, :3] = goal
    book[:3, 3] = fit["p_book_world"]
    desired[:3, 3] = fit["p_goal_world"]
    return book, desired
