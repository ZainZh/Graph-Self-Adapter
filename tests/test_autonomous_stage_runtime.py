import numpy as np
from scipy.spatial.transform import Rotation
import hashlib
import json
import io
from PIL import Image

from tools.deployment.autonomous_stage_runtime import AutonomousVision
from prove.xtrainer.perception import Observation
from tools.deployment.compact_stage_judge_v24 import CompactContext, JudgeError, OpenRouterJudge, SYSTEM_PROMPT, parse_reply
from tools.deployment.compact_stage_judge_v24_transition import SYSTEM_PROMPT as TRANSITION_PROMPT
from tools.deployment.autonomous_stage_runtime_box_support import supported_box_goal
from tools.deployment.autonomous_stage_runtime_white_move import white_move_vias
from tools.deployment.autonomous_stage_runtime_grasp_transition import public_history
from tools.deployment.autonomous_stage_runtime_initial_part import checked_handle_relation
from tools.deployment.autonomous_stage_runtime_object_yaw import object_centered_yaw, WHITE_YAW_DEG
from tools.deployment.compact_stage_judge_v24_grasp_transition import SYSTEM_PROMPT as GRASP_TRANSITION_PROMPT
import pytest


def test_proposal_roles_are_bound_before_actual_tracker_factory_and_review_clears_state():
    vision = AutonomousVision.__new__(AutonomousVision)
    vision.config = {"role_boxes": {"stale": [0, 0, 1, 1]}}
    vision.tracker = object()
    vision.latest_masks = {"stale": np.ones((2, 2), bool)}
    vision._memo = object()
    boxes = {
        "white_mug": [1, 1, 5, 5], "red_mug": [6, 1, 10, 5], "box": [11, 1, 15, 5],
        "white_handle": [2, 2, 4, 4], "table_patch": [1, 6, 5, 9],
    }
    seen = {}
    class Tracker:
        def initialize(self, _rgb, passed):
            seen["initialize"] = dict(passed)
            return {role: np.zeros((10, 16), bool) for role in passed}
    def factory():
        seen["factory_role_names"] = tuple(vision.config["role_boxes"])
        seen["old_masks_cleared"] = vision.latest_masks is None
        seen["old_memo_cleared"] = vision._memo is None
        return Tracker()
    vision._tracker = factory
    tracker, masks = vision._initialize_proposal_tracker(np.zeros((10, 16, 3), np.uint8), boxes)
    assert isinstance(tracker, Tracker)
    assert seen["factory_role_names"] == tuple(boxes)
    assert seen["initialize"] == boxes
    assert seen["old_masks_cleared"] and seen["old_memo_cleared"]
    assert set(masks) == set(boxes)


def test_actual_vision_init_creates_functional_evidence_directory_before_cloud_write(tmp_path):
    frame = tmp_path / "INITIAL_RGB.png"
    Image.fromarray(np.zeros((20, 20, 3), np.uint8)).save(frame, format="PNG")
    sha = hashlib.sha256(frame.read_bytes()).hexdigest()
    reply = tmp_path / "INITIAL_BOXES.json"
    reply.write_text(json.dumps({"frame_sha256": sha, "role_points": {"white_handle": [10, 10]}}))
    evidence = tmp_path / "run" / "SAM_MASK_EVIDENCE"
    config = {"live_view_root": str(tmp_path / "run" / "LIVE"),
              "prompt_frame_output": str(frame), "prompt_reply_path": str(reply),
              "role_boxes": {"white_mug": [4, 4, 16, 16], "white_handle": [6, 6, 15, 15],
                             "table_patch": [1, 1, 19, 19]}}
    class Camera:
        last_physical_time = None
        def close(self): pass
    vision = AutonomousVision(Camera(), np.eye(4), config, evidence_root=evidence)
    assert evidence.is_dir()
    vision.latest_depth = np.ones((20, 20), float)
    vision.latest_K = np.array([[100., 0., 10.], [0., 100., 10.], [0., 0., 1.]])
    mask = np.zeros((20, 20), bool); mask[6:15, 6:15] = True
    vision.latest_masks = {"white_mug": mask.copy(), "white_handle": mask,
                           "table_patch": np.zeros((20, 20), bool)}
    obs = Observation(1.0, np.zeros((20, 20, 3), np.uint8), {"white_mug": np.array([0., 0., 1.])}, "test", vision.latest_K)
    vision._record_initial_observation(obs)
    assert vision.latest_observation_stamp == 1.0 and vision.frame_count == 1
    assert (evidence / "MASK_EVIDENCE.json").is_file()
    cloud = vision.cloud("white_handle")
    assert len(cloud) == 81
    assert len(vision.cloud("white_mug")) == 81
    assert len(vision.cloud("table_patch")) >= 12
    assert (evidence / "ROOT_ASSOCIATED_HANDLE_COMPONENT.npz").is_file()
    assert (evidence / "ROOT_PART_ASSOCIATION.json").is_file()


def test_v24_positive_semantic_prompt_and_parser_require_first_and_latest_pair():
    assert 'exactly ["f0","f2"]' in SYSTEM_PROMPT
    compact = CompactContext({"requested_facts": ["holding"], "frames": [
        {"frame_id": "f0", "observed_monotonic_sec": 10.0},
        {"frame_id": "f2", "observed_monotonic_sec": 10.43},
    ]}, (), {})
    value = {"decision": "continue", "facts": {"holding": True},
             "evidence_frame_ids": ["f0", "f2"], "reason": "supported across first and latest frames"}
    assert parse_reply(value, compact)["facts"]["holding"] is True
    value["evidence_frame_ids"] = ["f2"]
    with pytest.raises(JudgeError, match="f0\\+f2"):
        parse_reply(value, compact)


def test_v24_records_exact_raw_model_response_before_reply_parse(tmp_path):
    compact = CompactContext({"requested_facts": ["holding"], "frames": [
        {"frame_id": "f0", "observed_monotonic_sec": 10.0},
        {"frame_id": "f2", "observed_monotonic_sec": 10.43},
    ]}, (), {"frames": [], "request_sha256": "request"})
    reply = {"decision": "continue", "facts": {"holding": True},
             "evidence_frame_ids": ["f0", "f2"], "reason": "joint evidence"}
    raw = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}], "usage": {"total_tokens": 3}}).encode()
    key = tmp_path / "key"; key.write_text("private-test-key")
    records = []
    judge = OpenRouterJudge(key, maximum_calls=1, opener=lambda *_a, **_k: io.BytesIO(raw), response_recorder=records.append)
    parsed, _ = judge.ask(compact)
    assert parsed["facts"]["holding"] is True and len(records) == 1
    assert records[0]["http_response_utf8"] == raw.decode() and "private-test-key" not in records[0]["http_response_utf8"]


def test_release_transition_prompt_distinguishes_sustained_holding_from_completion():
    assert "remain visibly held from f0 through f2" in TRANSITION_PROMPT
    assert "f0 may show the object still held" in TRANSITION_PROMPT
    assert "f2 must show the gripper open and retreated" in TRANSITION_PROMPT
    assert "object not stably at the target is not release or placement completion" in TRANSITION_PROMPT


def test_c02_box_support_goal_clamps_only_overhanging_axis_with_visible_margin():
    xx, yy = np.meshgrid(np.linspace(.10, .26, 20), np.linspace(-.64, -.48, 20))
    top = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, .14)]
    sides = np.c_[np.full(100, .10), np.linspace(-.64, -.48, 100), np.linspace(.02, .13, 100)]
    box = np.r_[top, sides]
    mx, my = np.meshgrid(np.linspace(-.14, -.06, 12), np.linspace(-.48, -.41, 12))
    mug = np.c_[mx.ravel(), my.ravel(), np.full(mx.size, .08)]
    safe, report = supported_box_goal(box, mug, [.172, -.634])
    assert abs(safe[0] - .172) < 1e-9
    assert safe[1] > -.602 and report["correction_xy_m"][1] > .03
    assert np.all(safe >= np.asarray(report["interior_lower_xy_m"]))
    assert np.all(safe <= np.asarray(report["interior_upper_xy_m"]))


def test_white_move_offline_continuation_has_two_vias_and_unchanged_final_target():
    current = np.array([0., 0., .26]); target = np.array([.12, -.18, .26])
    pad = np.array([0., 0., -.03]); rotation = np.eye(3)
    goals = white_move_vias(current, target, rotation, pad)
    handles = np.asarray([goal[0] + goal[1] @ pad for goal in goals])
    assert np.allclose(handles[0], current * (2 / 3) + target / 3)
    assert np.allclose(handles[1], current / 3 + target * (2 / 3))
    assert np.allclose(handles[2], target)
    assert np.allclose(handles[:, 2], target[2])


def test_grasp_transition_prompt_allows_only_supported_partial_occlusion():
    assert "actual current pre-close stage-entry frame" in GRASP_TRANSITION_PROMPT
    assert "Partial object occlusion by the post-close jaws is permitted only" in GRASP_TRANSITION_PROMPT
    assert "not proof of attachment or measured contact" in GRASP_TRANSITION_PROMPT


def test_public_history_uses_actual_sample_fields():
    class Sample:
        q=np.array([1.,2.]); qdot=np.array([.1,.2]); tcp=np.array([.3,.4,.5]); stamp=12.3
    assert public_history([Sample()]) == [{"q":[1.,2.],"qdot":[.1,.2],"TCP_m":[.3,.4,.5],"stamp":12.3}]


def test_initial_part_gate_reports_unchanged_narrow_tube_metrics():
    z=np.linspace(-.02,.02,30); handle=np.c_[np.zeros(30),np.zeros(30),z]
    cup=np.c_[np.linspace(-.03,.03,30),np.full(30,.02),np.linspace(0,.06,30)]
    relation,metrics=checked_handle_relation(handle,np.array([0.,-.5,.3]),cup)
    assert metrics["thresholds"] == {"minimum_span_m":.008,"minimum_principal_ratio":1.5,"maximum_radial_q90_m":.016}
    assert metrics["span_m"] >= .008 and relation["radial_extent_m"] <= .016


def test_white_object_yaw_preserves_cup_center_and_upright_axis():
    handle=np.array([.08,-.52,.26]);delta=np.array([-.055,0.,.006]);rotation=np.eye(3)
    turned,R,cup=object_centered_yaw(handle,delta,rotation)
    assert WHITE_YAW_DEG == -30.0
    assert np.allclose(turned - Rotation.from_euler("z",WHITE_YAW_DEG,degrees=True).as_matrix() @ delta,cup)
    assert np.allclose(cup,handle-delta)
    assert np.allclose(R[:,2],[0.,0.,1.])
