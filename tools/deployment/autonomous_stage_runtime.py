"""V24 shared runtime: V23 control plus read-only current RGB/stage publication."""
import json
from pathlib import Path
import os
import hashlib
import io

from PIL import Image
import numpy as np

from all_code_stage_runtime import AllCodeStageExecutor
from all_code_stage_runtime import SharedSAMVision as BaseVision
from autonomous_sam_grounding import ground_initial_frame
from handle_local_candidate_v4 import observed_patch_cloud, seed_depth_component, tube_relation
from prove.xtrainer.io import SafetyError
from prove.xtrainer.perception import Observation, role_geometry
from prove.xtrainer.agent import digest


VARIANT = "cup_autonomous_stage_v24"


class AutonomousVision(BaseVision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Base entry has already claimed/created the fresh run output before
        # vision construction. Functional-part evidence may be written by
        # cloud() before the inherited sparse-frame recorder runs.
        if self.evidence_root is not None: Path(self.evidence_root).mkdir(parents=True, exist_ok=True)
        self.live_root = Path(self.config["live_view_root"])
        self.live_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _table_plane(depth, K, T, box):
        """Metric RGBD ROI plane; table is deliberately not a SAM instance."""
        x0, y0, x1, y1 = (int(v) for v in box)
        h, w = depth.shape[:2]
        x0, x1 = max(0, x0), min(w, x1); y0, y1 = max(0, y0), min(h, y1)
        yy, xx = np.mgrid[y0:y1, x0:x1]; z = np.asarray(depth[y0:y1, x0:x1], float)
        valid = np.isfinite(z) & (z > 0)
        if np.count_nonzero(valid) < 12: raise SafetyError("table RGBD ROI has fewer than 12 valid depth pixels")
        zv, xv, yv = z[valid], xx[valid], yy[valid]
        camera = np.c_[((xv - K[0, 2]) * zv / K[0, 0]), ((yv - K[1, 2]) * zv / K[1, 1]), zv]
        world = camera @ np.asarray(T[:3, :3]).T + np.asarray(T[:3, 3])
        center = np.median(world, axis=0); _, _, vh = np.linalg.svd(world - center, full_matrices=False)
        normal = vh[-1]; residual = np.abs((world - center) @ normal)
        inliers = residual <= 0.003
        cloud = world[inliers]
        if len(cloud) < 12 or len(cloud) < .8 * len(world): raise SafetyError("table RGBD ROI does not support a 3mm plane")
        if abs(float(normal[2])) < .9: raise SafetyError("table RGBD ROI plane normal is not horizontal in robot frame")
        if np.ptp(cloud[:, 0]) < .01 or np.ptp(cloud[:, 1]) < .01: raise SafetyError("table RGBD ROI plane extent is unresolved")
        mask = np.zeros((h, w), bool); selected_y, selected_x = yv[inliers], xv[inliers]; mask[selected_y, selected_x] = True
        return cloud, mask

    def cloud(self, role):
        if role == "table_patch":
            cloud, _ = self._table_plane(self.latest_depth, self.latest_K, self.T, self.config["role_boxes"][role])
            return cloud
        return super().cloud(role)

    def _initialize_proposal_tracker(self, rgb, boxes):
        # A rejected proposal must not leak tracker/config/cache state into the
        # one allowed evidence-based review proposal.
        self.tracker = None
        self.config["role_boxes"] = {}
        self.latest_masks = None
        if hasattr(self, "_memo"): self._memo = None
        self.config["role_boxes"] = {role: list(box) for role, box in boxes.items()}
        tracker = self._tracker()
        masks = tracker.initialize(rgb, self.config["role_boxes"])
        if set(masks) != set(boxes): raise SafetyError("SAM tracker placeholders do not match five proposed roles")
        return tracker, masks

    def _publish(self, obs):
        frame = self.live_root / "LATEST_RGB.jpg"
        temporary = self.live_root / ".LATEST_RGB.jpg.tmp"
        Image.fromarray(obs.rgb).save(temporary, format="JPEG", quality=85); os.replace(temporary, frame)
        stage = self.stage_context() or {}
        status = {"stage_id": stage.get("stage_id"), "phase": stage.get("phase"),
                  "observation_stamp": obs.stamp, "frame_index": getattr(self, "frame_count", None),
                  "source": obs.source, "read_only_view": True}
        target = self.live_root / "LIVE_STATUS.json"; tmp = self.live_root / ".LIVE_STATUS.json.tmp"
        tmp.write_text(json.dumps(status, indent=2) + "\n"); os.replace(tmp, target)

    def _record_initial_observation(self, obs):
        self.latest_observation_stamp = obs.stamp
        index = self.frame_count
        root = Path(self.evidence_root); root.mkdir(parents=True, exist_ok=True)
        rgb_path = root / f"RGB_{index:06d}.png"; Image.fromarray(obs.rgb).save(rgb_path)
        masks_path = root / f"MASKS_{index:06d}.npz"; np.savez_compressed(masks_path, **self.latest_masks)
        rgbd_path = root / f"RGBD_{index:06d}.npz"
        np.savez_compressed(rgbd_path, depth_m=self.latest_depth, K=self.latest_K, T=self.T)
        files = [{"path": str(path), "sha256": digest(path)} for path in (rgb_path, masks_path, rgbd_path)]
        points = {key: None if value is None else np.asarray(value).tolist() for key, value in obs.roles.items()}
        self.mask_records.append({"frame_index": index, "stamp": obs.stamp,
            "physical_time_sec": self.raw_camera.last_physical_time, "role_points": points,
            "files": files, "mask_source": "actual_SAM2", "GT_fallback": False})
        context = self.stage_context() or {}
        self.frame_records.append({"frame_index": index, "stage_id": context.get("stage_id"), "phase": context.get("phase"),
            "observed_monotonic_sec": obs.stamp, "recorded_monotonic_sec": self.clock(),
            "physical_time_sec": self.raw_camera.last_physical_time,
            "clock_source": "simulator camera snapshot physical clock" if self.raw_camera.last_physical_time is not None else "real camera monotonic wall timestamp; physical clock unavailable",
            "mask_pixels": {key: int(np.asarray(value, bool).sum()) for key, value in self.latest_masks.items()},
            "mask_source": "actual_SAM2", "GT_fallback": False, "files": files, "role_points": points})
        self.frame_count += 1
        self.save_frame_evidence()

    def _initialize_autonomously(self):
        stamp, rgb, depth, K = self.camera.capture()
        buffer = io.BytesIO(); Image.fromarray(rgb).save(buffer, format="PNG"); raw = buffer.getvalue()
        frame = Path(self.config["prompt_frame_output"]); frame.parent.mkdir(parents=True, exist_ok=True); frame.write_bytes(raw)
        expected = hashlib.sha256(raw).hexdigest(); accepted = {}
        roles = ("white_mug", "red_mug", "box", "white_handle", "table_patch")
        def diagnostics(masks=None, valid=False, reason=None, seed_valid=False, seed_in_roi=False):
            masks = masks or {}
            pixels = {role: int(np.count_nonzero(masks.get(role, np.zeros(depth.shape, bool)))) for role in roles}
            depths = {role: int(np.count_nonzero(np.asarray(masks.get(role, np.zeros(depth.shape, bool)), bool) & np.isfinite(depth) & (depth > 0))) for role in roles}
            boxes = {}
            for role in roles:
                ys, xs = np.nonzero(np.asarray(masks.get(role, np.zeros(depth.shape, bool)), bool))
                boxes[role] = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            return {"valid": bool(valid), "retry_reason": None if valid else str(reason)[:240],
                    "role_pixel_count": pixels, "role_depth_count": depths,
                    "white_handle_seed_valid": bool(seed_valid), "seed_in_roi": bool(seed_in_roi), "segment_bbox": boxes,
                    "role_source": {"white_mug": "actual_SAM2", "red_mug": "actual_SAM2", "box": "actual_SAM2",
                                    "white_handle": "actual_SAM2_seed_depth_component", "table_patch": "RGBD_ROI_plane"}}
        def validate(payload):
            if payload.get("frame_sha256") != expected: return diagnostics(reason="proposal is not bound to the current RGB")
            boxes = payload.get("role_boxes", {}); seed = payload.get("role_points", {}).get("white_handle")
            if set(boxes) != set(roles): return diagnostics(reason="five required roles are incomplete")
            masks = None; seed_supported = False
            seed_in_roi = bool(seed and all(boxes[role][0] <= seed[0] < boxes[role][2] and boxes[role][1] <= seed[1] < boxes[role][3]
                                                for role in ("white_handle", "white_mug")))
            try:
                tracker, masks = self._initialize_proposal_tracker(rgb, boxes)
                centers = role_geometry(depth, masks, K, self.T)
                if any(centers.get(role) is None for role in ("white_mug", "red_mug", "box")): return diagnostics(masks, reason="whole-object SAM mask has insufficient valid depth")
                component = seed_depth_component(depth, masks["white_handle"], K, boxes["white_handle"], seed)
                seed_supported = True
                handle = observed_patch_cloud(depth, component, K, self.T, boxes["white_handle"])
                table, table_mask = self._table_plane(depth, K, self.T, boxes["table_patch"])
                diagnostic_masks = dict(masks); diagnostic_masks["table_patch"] = table_mask
                tube_relation(handle, self.T[:3, 3])
            except Exception as error:
                return diagnostics(locals().get("diagnostic_masks", masks), reason=error,
                                   seed_valid=seed_supported, seed_in_roi=seed_in_roi)
            accepted.update(tracker=tracker, masks=masks, centers=centers)
            return diagnostics(diagnostic_masks, valid=True, seed_valid=True, seed_in_roi=seed_in_roi)
        payload = ground_initial_frame(frame, Path(self.config["prompt_reply_path"]),
            self.config["autonomous_grounding_key_file"], maximum_calls=self.config["autonomous_grounding_maximum_calls"],
            semantic_validator=validate, evidence_root=Path(self.config["autonomous_grounding_evidence_root"]))
        if payload.get("frame_sha256") != expected or "tracker" not in accepted: raise SafetyError("autonomous grounding did not produce validated current-frame roles")
        self.config["role_boxes"] = payload["role_boxes"]; self.tracker = accepted["tracker"]
        # Initialization/API latency never refreshes the old stamp. Resume from a
        # truly new camera frame and retain the original freshness contract.
        stamp, rgb, depth, K = self.camera.capture(); masks = self.tracker.track(rgb)
        if not 0 <= self.clock() - stamp <= self.config["max_age_sec"]: raise SafetyError("RGBD/SAM processing too stale")
        obs = Observation(stamp, rgb, role_geometry(depth, masks, K, self.T), "RGBD+actual_SAM2+autonomous_Gemini_grounding", K)
        self._record_initial_observation(obs)
        return obs

    def capture(self):
        obs = self._initialize_autonomously() if self.tracker is None else super().capture()
        self._publish(obs)
        return obs


class AutonomousExecutor(AllCodeStageExecutor):
    def run(self):
        result = super().run()
        result.update(execution_variant=VARIANT, autonomous_initial_grounding=True,
                      manual_grounding_requests=0, live_view_is_control_input=False,
                      table_patch_source="same-frame metric RGBD ROI verified plane; not SAM or simulator truth")
        return result


SharedCupExecutor = AutonomousExecutor
SharedSAMVision = AutonomousVision
