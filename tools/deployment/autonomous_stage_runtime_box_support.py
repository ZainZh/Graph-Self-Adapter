"""V24 autonomous runtime with a bounded c02 box-top support correction."""
import numpy as np

from autonomous_stage_runtime import AutonomousExecutor, AutonomousVision
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError


VARIANT = "cup_autonomous_stage_v24_box_support"


def supported_box_goal(box_cloud, mug_cloud, nominal_xy):
    box = np.asarray(box_cloud, float); mug = np.asarray(mug_cloud, float); nominal = np.asarray(nominal_xy, float)
    if box.ndim != 2 or box.shape[1] != 3 or len(box) < 50 or mug.ndim != 2 or mug.shape[1] != 3 or len(mug) < 50:
        raise SafetyError("current SAM/RGBD box or mug support geometry insufficient")
    # Visible top points occupy the upper half of the box cloud in the shared
    # gravity frame. Robust quantiles reject mask fringes and side-face noise.
    top = box[box[:, 2] >= np.quantile(box[:, 2], .5)]
    lower, upper = np.quantile(top[:, :2], (.02, .98), axis=0)
    mug_lower, mug_upper = np.quantile(mug[:, :2], (.1, .9), axis=0)
    margin = .5 * (mug_upper - mug_lower) + .01
    interior_lower, interior_upper = lower + margin, upper - margin
    if np.any(interior_lower >= interior_upper): raise SafetyError("observed box top cannot support bounded mug footprint")
    safe = np.clip(nominal, interior_lower, interior_upper)
    return safe, {"box_top_lower_xy_m": lower.tolist(), "box_top_upper_xy_m": upper.tolist(),
                  "visible_mug_half_extent_xy_m": (.5 * (mug_upper - mug_lower)).tolist(),
                  "support_margin_xy_m": margin.tolist(), "interior_lower_xy_m": interior_lower.tolist(),
                  "interior_upper_xy_m": interior_upper.tolist(), "nominal_object_goal_xy_m": nominal.tolist(),
                  "safe_object_goal_xy_m": safe.tolist(), "correction_xy_m": (safe - nominal).tolist(),
                  "source": "current actual SAM masks plus metric RGBD; no simulator truth or continuous object pose"}


class BoxSupportExecutor(AutonomousExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self._c02_support_applied = False; self.box_support_plan = None

    def _ground_red_cycle(self, stage, obs):
        super()._ground_red_cycle(stage, obs)
        if stage["cycle_id"] != "c02" or self._c02_support_applied: return
        align = next(row for row in self.b["graph"]["stages"] if row["cycle_id"] == "c02" and row["phase"] == "Align")
        reference = self.nearest_ref(float(align["reference"]["end_time_sec"]))["surface_centroids_world_m"].get(stage["object"])
        if reference is None: raise SafetyError("c02 declared Align object reference missing")
        nominal = np.asarray(reference)[:2] + self.coarse_target[:2]
        safe, report = supported_box_goal(self.vision.cloud("box"), self.vision.cloud(stage["object"]), nominal)
        correction = safe - nominal
        self.coarse_target[:2] += correction
        self.target_anchor[:2] += correction
        self.static_target_env = self.coarse_target.copy(); self.code_static_target_handled = self.coarse_target.copy()
        self.target_prior += "; c02 object goal clamped into current visible box-top support interior"
        self._c02_support_applied = True; self.box_support_plan = report
        write(self.output / "C02_BOX_SUPPORT_PLAN.json", report)

    def run(self):
        result = super().run()
        result.update(execution_variant=VARIANT, c02_box_support_plan=self.box_support_plan,
                      c02_box_support_source="actual SAM/RGBD visible geometry; no private truth")
        return result


SharedCupExecutor = BoxSupportExecutor
SharedSAMVision = AutonomousVision
