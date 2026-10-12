"""V24 validates the actual functional handle geometry before any stage motion."""
from pathlib import Path
import hashlib
import io

import numpy as np
from PIL import Image

from autonomous_stage_runtime_grasp_transition import GraspTransitionExecutor
from autonomous_stage_runtime import AutonomousVision
from autonomous_sam_grounding import ground_initial_frame
from handle_local_candidate_v4 import handle_plane_relation, observed_patch_cloud, seed_depth_component
from prove.xtrainer.io import SafetyError
from prove.xtrainer.perception import Observation, role_geometry

VARIANT = "cup_autonomous_stage_v24_initial_part_contract"


def checked_handle_relation(handle, camera, cup):
    points = np.asarray(handle, float); center = np.median(points, axis=0)
    _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
    tangent = vectors[0]; span = float(np.ptp((points - center) @ tangent))
    radial = (points - center) - np.outer((points - center) @ tangent, tangent)
    extent = float(np.quantile(np.linalg.norm(radial, axis=1), .9))
    ratio = float(singular[0] / max(singular[1], 1e-9))
    metrics = {"span_m": span, "principal_ratio": ratio, "radial_q90_m": extent,
               "thresholds": {"minimum_span_m": .008, "minimum_principal_ratio": 1.5, "maximum_radial_q90_m": .016}}
    try:
        relation = handle_plane_relation(points, camera, cup)
    except Exception as error:
        raise SafetyError(f"{error}; measured span={span:.6f}m ratio={ratio:.4f} radial_q90={extent:.6f}m; select a narrower local visible outer handle-tube patch")
    return relation, metrics


class RootFallbackGroundingVision(AutonomousVision):
    def _initialize_autonomously(self):
        self._actual_root_provider=False
        try:return self._initialize_selected_provider()
        except SafetyError as error:
            if 'automatic grounding remained invalid/unknown' not in str(error):raise
            self._actual_root_provider=True
            self.config['prompt_frame_output']=str(Path(self.config['prompt_frame_output']).with_name('INITIAL_ROOT_RGB.png'))
            self.config['prompt_reply_path']=str(Path(self.config['prompt_reply_path']).with_name('INITIAL_ROOT_BOXES.json'))
            return self._initialize_selected_provider()

    def _initialize_selected_provider(self):
        _, rgb, depth, K = self.camera.capture()
        buffer = io.BytesIO(); Image.fromarray(rgb).save(buffer, format="PNG"); raw = buffer.getvalue()
        frame = Path(self.config["prompt_frame_output"]); frame.parent.mkdir(parents=True, exist_ok=True); frame.write_bytes(raw)
        expected = hashlib.sha256(raw).hexdigest(); accepted = {}
        roles = ("white_mug", "red_mug", "box", "white_handle", "table_patch")

        def diagnostics(local_depth, masks=None, valid=False, reason=None, seed_valid=False, seed_in_roi=False):
            masks = masks or {}; shape = local_depth.shape
            pixels = {role: int(np.count_nonzero(masks.get(role, np.zeros(shape, bool)))) for role in roles}
            depths = {role: int(np.count_nonzero(np.asarray(masks.get(role, np.zeros(shape, bool)), bool)
                                                & np.isfinite(local_depth) & (local_depth > 0))) for role in roles}
            bboxes = {}
            for role in roles:
                ys, xs = np.nonzero(np.asarray(masks.get(role, np.zeros(shape, bool)), bool))
                bboxes[role] = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            return {"valid": bool(valid), "retry_reason": None if valid else str(reason)[:240],
                    "role_pixel_count": pixels, "role_depth_count": depths,
                    "white_handle_seed_valid": bool(seed_valid), "seed_in_roi": bool(seed_in_roi),
                    "segment_bbox": bboxes,
                    "role_source": {"white_mug": "actual_SAM2", "red_mug": "actual_SAM2", "box": "actual_SAM2",
                                    "white_handle": "actual_SAM2_seed_depth_component", "table_patch": "RGBD_ROI_plane"}}

        def evaluate(local_rgb, local_depth, local_K, boxes, seed, tracker=None, initialize=False):
            if initialize:
                tracker, masks = self._initialize_proposal_tracker(local_rgb, boxes)
            else:
                masks = tracker.track(local_rgb)
            centers = role_geometry(local_depth, masks, local_K, self.T)
            if any(centers.get(role) is None for role in ("white_mug", "red_mug", "box")):
                raise SafetyError("whole-object SAM mask has insufficient valid depth")
            component = seed_depth_component(local_depth, masks["white_handle"], local_K, boxes["white_handle"], seed)
            handle = observed_patch_cloud(local_depth, component, local_K, self.T, boxes["white_handle"])
            cup = observed_patch_cloud(local_depth, masks["white_mug"], local_K, self.T, boxes["white_mug"])
            # The downstream controller consumes this exact handle-plane relation.
            # Validate it here with the unchanged predicate rather than after red motion.
            relation, part_metrics = checked_handle_relation(handle, self.T[:3, 3], cup)
            table, table_mask = self._table_plane(local_depth, local_K, self.T, boxes["table_patch"])
            diagnostic_masks = dict(masks); diagnostic_masks["table_patch"] = table_mask
            return tracker, masks, centers, diagnostic_masks, component, part_metrics

        def validate(payload):
            if payload.get("frame_sha256") != expected:
                return diagnostics(depth, reason="proposal is not bound to the current RGB")
            boxes = payload.get("role_boxes", {}); seed = payload.get("role_points", {}).get("white_handle")
            if set(boxes) != set(roles): return diagnostics(depth, reason="five required roles are incomplete")
            seed_in_roi = bool(seed and all(boxes[role][0] <= seed[0] < boxes[role][2] and boxes[role][1] <= seed[1] < boxes[role][3]
                                                for role in ("white_handle", "white_mug")))
            try:
                tracker, _, _, _, _, _ = evaluate(rgb, depth, K, boxes, seed, initialize=True)
                stamp2, rgb2, depth2, K2 = self.camera.capture()
                tracker, masks2, centers2, diagnostic2, _, part_metrics2 = evaluate(rgb2, depth2, K2, boxes, seed, tracker=tracker)
            except Exception as error:
                reason = ("white_handle must select the visible outer mug-handle tube, not cup rim/body; "
                          "actual downstream geometry rejected it: " + str(error))
                return diagnostics(locals().get("depth2", depth), locals().get("diagnostic2", locals().get("masks2")),
                                   reason=reason, seed_valid=False, seed_in_roi=seed_in_roi)
            accepted.update(tracker=tracker, masks=masks2, centers=centers2, stamp=stamp2,
                            rgb=rgb2, depth=depth2, K=K2, part_metrics=part_metrics2)
            return diagnostics(depth2, diagnostic2, valid=True, seed_valid=True, seed_in_roi=seed_in_roi)

        provider=ground_initial_frame
        if self._actual_root_provider:
            from cup_v25_root_grounding_bridge import ground_initial_frame as provider
        payload = provider(frame, Path(self.config["prompt_reply_path"]),
            self.config["autonomous_grounding_key_file"], maximum_calls=self.config["autonomous_grounding_maximum_calls"],
            semantic_validator=validate, evidence_root=Path(self.config["autonomous_grounding_evidence_root"]))
        if payload.get("frame_sha256") != expected or "tracker" not in accepted:
            raise SafetyError("autonomous grounding did not produce validated current-frame roles")
        if not 0 <= self.clock() - accepted["stamp"] <= self.config["max_age_sec"]:
            raise SafetyError("accepted functional-part RGBD/SAM observation stale")
        self.config["role_boxes"] = payload["role_boxes"]; self.tracker = accepted["tracker"]
        self.latest_depth = accepted["depth"]; self.latest_K = accepted["K"]; self.latest_masks = accepted["masks"]
        self.initial_part_metrics = accepted["part_metrics"]
        obs = Observation(accepted["stamp"], accepted["rgb"], accepted["centers"],
                          "RGBD+actual_SAM2+autonomous_Gemini_grounding+functional_part_gate", accepted["K"])
        self._record_initial_observation(obs)
        from prove.xtrainer.agent import write
        write(Path(self.config['autonomous_grounding_evidence_root'])/'INITIAL_GROUNDING_SOURCE.json',{
            'source':'actual current Codex Root with same actual SAM/RGBD/tube/plane validation' if self._actual_root_provider else 'Gemini automatic with actual SAM/RGBD/tube/plane validation',
            'current_frame':str(frame),'frame_sha256':expected,'fresh_observation_stamp':obs.stamp})
        return obs


class InitialPartContractExecutor(GraspTransitionExecutor):
    def run(self):
        result = super().run()
        result.update(execution_variant=VARIANT,
                      initial_functional_part_contract="same downstream narrow-tube and handle-plane predicates before motion",
                      initial_functional_part_metrics=getattr(self.vision, "initial_part_metrics", None))
        return result

SharedCupExecutor = InitialPartContractExecutor
SharedSAMVision = RootFallbackGroundingVision
