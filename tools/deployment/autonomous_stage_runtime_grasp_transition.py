"""V24 current pre-close/post-close Grasp evidence; control remains unchanged."""
import numpy as np
from autonomous_stage_runtime_white_move import WhiteMoveContinuationExecutor
from autonomous_stage_runtime import AutonomousVision
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError

VARIANT = "cup_autonomous_stage_v24_grasp_transition_evidence"


def public_history(samples):
    return [{"q": s.q.tolist(), "qdot": s.qdot.tolist(), "TCP_m": s.tcp.tolist(), "stamp": s.stamp}
            for s in samples]


class GraspTransitionExecutor(WhiteMoveContinuationExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grasp_entry_frames = {}

    def capture(self):
        samples, obs = super().capture()
        stage = self.stage
        if stage is not None and stage.get("phase") == "Grasp" and stage["stage_id"] not in self.grasp_entry_frames:
            if obs.roles.get(stage["object"]) is None:
                raise SafetyError("current Grasp-entry object is not visible")
            row = (obs.stamp, obs.rgb.copy(), public_history(samples))
            self.grasp_entry_frames[stage["stage_id"]] = row
            write(self.output / "GRASP_ENTRY_EVIDENCE.json", {
                key: {"observed_monotonic_sec": value[0], "source": "current live RGB at Grasp entry before close",
                      "object_visible": True, "not_demo_or_restored_prefix": True}
                for key, value in self.grasp_entry_frames.items()})
        return samples, obs

    def semantics(self, stage, facts, before):
        if stage.get("phase") == "Grasp" and "holding" in facts:
            entry = self.grasp_entry_frames.get(stage["stage_id"])
            if entry is None:
                raise SafetyError("current pre-close Grasp-entry evidence missing")
            samples, obs = before
            current = list(obs.history or [])
            latest = (obs.stamp, obs.rgb.copy(), public_history(samples))
            candidates = [row for row in current if entry[0] < row[0] < latest[0]]
            if latest[0] - entry[0] < .19 or not candidates:
                raise SafetyError("current Grasp transition lacks ordered f0/mid/f2 evidence")
            middle = candidates[len(candidates) // 2]
            obs.history = [entry, middle, latest]
            write(self.output / "GRASP_TRANSITION_EVIDENCE.json", {
                "stage_id": stage["stage_id"], "frame_stamps": [entry[0], middle[0], latest[0]],
                "f0_source": "current Grasp entry before close", "f2_source": "fresh post-close current frame",
                "span_sec": latest[0] - entry[0], "object_visible_at_entry": True,
                "permission_scope": "tentative_macro_grasp; not attachment/contact proof"})
        return super().semantics(stage, facts, before)

    def run(self):
        result = super().run()
        result.update(execution_variant=VARIANT,
                      grasp_transition_evidence="current pre-close entry plus fresh post-close; no demo/restored frame")
        return result

SharedCupExecutor = GraspTransitionExecutor
SharedSAMVision = AutonomousVision
