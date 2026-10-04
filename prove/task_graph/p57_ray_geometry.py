"""P57 public-ray reconstruction of predicted virtual anchors.

Pure geometry. No evaluator truth, gradients, or adaptation.
"""
import numpy as np
from prove.task_graph.p57_public_geometry import rays

NOMINAL = np.array([[0., 0., 0.], [.03, 0., 0.], [0., 0., .03]])
ROLES = ("book", "desired")


def _check_pose(pose, name):
    if not np.isfinite(pose).all() or not np.allclose(pose[..., 3, :], [0., 0., 0., 1.]):
        raise ValueError(f"Invalid {name} transform")
    rotation = pose[..., :3, :3]
    if not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"Nonorthogonal {name} rotation")
    if np.any(np.linalg.det(rotation) < .999):
        raise ValueError(f"Reflected {name} rotation")


def _inputs(pred_uv, role_valid, K, cam_world, eef_world, history, cond_limit):
    uv = np.asarray(pred_uv, dtype=np.float64)
    valid = np.asarray(role_valid)
    K = np.asarray(K, dtype=np.float64)
    camera = np.asarray(cam_world, dtype=np.float64)
    eef = np.asarray(eef_world, dtype=np.float64)
    if uv.ndim != 5 or uv.shape[1:] != (2, 2, 3, 2) or uv.shape[0] < 1:
        raise ValueError("pred_uv must have shape [T,2,2,3,2]")
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


def _fit(origins, directions, cond_limit):
    n = len(origins)
    if n < 2:
        return None, dict(rank=0 if n == 0 else 2, condition=None,
                          rms_ray_m=None, rays=n, reason="insufficient_views")
    P = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    A = P.sum(axis=0)
    eig = np.linalg.eigvalsh(A)
    # Match the Stage A two-view readout's absolute numerical singular gate.
    rank = int(np.count_nonzero(eig >= 1e-5))
    condition = float(eig[-1] / eig[0]) if eig[0] > 0 else None
    diagnostic = dict(rank=rank, condition=condition, rms_ray_m=None, rays=n, reason=None)
    if rank < 3:
        diagnostic["reason"] = "rank_deficient"
        return None, diagnostic
    if condition is None or condition > cond_limit:
        diagnostic["reason"] = "ill_conditioned"
        return None, diagnostic
    point = np.linalg.solve(A, np.einsum("nij,nj->i", P, origins))
    if np.any(np.einsum("ni,ni->n", point - origins, directions) <= 0):
        diagnostic["reason"] = "behind_camera"
        return None, diagnostic
    residual = np.einsum("nij,nj->ni", P, point - origins)
    diagnostic["rms_ray_m"] = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return point, diagnostic


def _frame(points):
    """Nearest proper rigid frame; retain raw-anchor deformation separately."""
    src = NOMINAL - NOMINAL.mean(axis=0)
    dst = points - points.mean(axis=0)
    shape_singular = np.linalg.svd(dst, compute_uv=False)
    shape_rank = int(np.count_nonzero(shape_singular > max(shape_singular[0] * 1e-6, 1e-8)))
    if shape_rank < 2:
        return None, None, None, shape_rank, shape_singular
    U, _, Vt = np.linalg.svd(src.T @ dst)
    D = np.eye(3)
    D[-1, -1] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T
    t = points.mean(axis=0) - R @ NOMINAL.mean(axis=0)
    fitted = NOMINAL @ R.T + t
    frame = np.eye(4)
    frame[:3, :3], frame[:3, 3] = R, t
    error = float(np.sqrt(np.mean(np.sum((points - fitted)**2, axis=1))))
    return frame, fitted, error, shape_rank, shape_singular


def _window(t, first, uv, valid, K, camera, eef, cond_limit):
    out = {key: {} for key in ("valid", "anchor_valid", "rank", "condition", "rms_ray_m",
                                "reason", "rays", "deformation_rms_m", "frame_rank",
                                "frame_singular_values", "frame_reason")}
    for r, role in enumerate(ROLES):
        points, diagnostics = [], []
        for anchor in range(3):
            origins, directions = [], []
            for s in range(first, t + 1):
                for view in range(2):
                    if not valid[s, view, r]:
                        continue
                    origin, direction = rays(uv[s, view, r, anchor], K[s, view], camera[s, view])
                    if r == 0:  # Book point is fixed in EEF coordinates.
                        Rs, Rt = eef[s, :3, :3], eef[t, :3, :3]
                        origin = Rt @ Rs.T @ (origin - eef[s, :3, 3]) + eef[t, :3, 3]
                        direction = Rt @ Rs.T @ direction
                    origins.append(origin)
                    directions.append(direction)
            point, diag = _fit(np.asarray(origins).reshape(-1, 3),
                               np.asarray(directions).reshape(-1, 3), cond_limit)
            points.append(point)
            diagnostics.append(diag)
        anchors_valid = all(point is not None for point in points)
        if anchors_valid:
            anchors = np.stack(points)
            frame, rigid_anchors, deformation, frame_rank, singular = _frame(anchors)
            frame_reason = None if frame is not None else "frame_collinear"
        else:
            anchors = frame = rigid_anchors = deformation = None
            frame_rank = singular = None
            frame_reason = "missing_anchor"
        out[f"{role}_world_anchors"] = anchors
        out[f"{role}_frame"] = frame
        out[f"{role}_rigid_anchors"] = rigid_anchors
        out["anchor_valid"][role] = anchors_valid
        out["valid"][role] = frame is not None
        out["deformation_rms_m"][role] = deformation
        out["frame_rank"][role] = frame_rank
        out["frame_singular_values"][role] = None if singular is None else singular.tolist()
        out["frame_reason"][role] = frame_reason
        for key in ("rank", "condition", "rms_ray_m", "reason", "rays"):
            out[key][role] = [d[key] for d in diagnostics]
    return out


def fit_virtual_anchor_frames(pred_uv, role_valid, K, cam_world, eef_world,
                              *, history=8, cond_limit=1e5):
    """Return instant and history public-ray fits for each time step.

    Book history is transformed through each frame's EEF into the current EEF;
    desired-book history stays fixed in world. Missing/degenerate rays return
    None anchors and frames with explicit rank, condition and reasons.
    """
    uv, valid, K, camera, eef = _inputs(
        pred_uv, role_valid, K, cam_world, eef_world, history, cond_limit)
    return [dict(instant=_window(t, t, uv, valid, K, camera, eef, cond_limit),
                 history=_window(t, max(0, t - history + 1), uv, valid,
                                 K, camera, eef, cond_limit))
            for t in range(len(uv))]
