import io
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.deployment.autonomous_sam_grounding import (
    DEFAULT_CUP_ROLES,
    GroundingError,
    GroundingUnknown,
    OpenRouterGrounder,
    ROLE_SOURCES,
    ground_initial_frame,
    normalized_proposal_to_pixels,
    validate_semantic_diagnostics,
    validate_proposal,
)


VALID = {
    "status": "proposal",
    "role_boxes": {
        "white_mug": [10, 10, 60, 80],
        "red_mug": [70, 10, 120, 80],
        "box": [125, 10, 175, 75],
        "white_handle": [20, 25, 45, 55],
        "table_patch": [5, 82, 65, 98],
    },
    "role_points": {"white_handle": [30, 35]},
    "reason": "visible solid regions selected",
}


class Response:
    def __init__(self, value):
        self.raw = json.dumps(
            {"choices": [{"message": {"content": json.dumps(value)}}], "usage": {"total_tokens": 20}}
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.raw


def frame(path: Path):
    Image.new("RGB", (180, 100), (80, 90, 100)).save(path, format="PNG")


def private_key(path: Path):
    path.write_text("test-key-value", encoding="utf-8")
    path.chmod(0o600)


def client(tmp_path, replies):
    key = tmp_path / "key"
    private_key(key)
    queue = list(replies)
    return OpenRouterGrounder(key, maximum_calls=len(queue), opener=lambda *_a, **_k: Response(queue.pop(0)))


def diagnostics(valid=True, reason=None):
    return {
        "valid": valid,
        "retry_reason": reason,
        "role_pixel_count": {role: (20 if valid else (0 if role == "table_patch" else 20)) for role in DEFAULT_CUP_ROLES},
        "role_depth_count": {role: (18 if valid else (0 if role == "table_patch" else 18)) for role in DEFAULT_CUP_ROLES},
        "white_handle_seed_valid": True,
        "seed_in_roi": True,
        "segment_bbox": {role: ([1, 1, 5, 5] if valid or role != "table_patch" else None) for role in DEFAULT_CUP_ROLES},
        "role_source": dict(ROLE_SOURCES),
    }


def test_complete_numeric_proposal_and_seed_geometry():
    boxes, points, _ = validate_proposal(VALID, width=180, height=100)
    assert tuple(boxes) == tuple(DEFAULT_CUP_ROLES)
    assert points["white_handle"] == [30, 35]
    partial = {**VALID, "role_boxes": {k: v for k, v in VALID["role_boxes"].items() if k != "table_patch"}}
    with pytest.raises(GroundingError, match="five"):
        validate_proposal(partial, width=180, height=100)
    outside = json.loads(json.dumps(VALID))
    outside["role_points"]["white_handle"] = [150, 90]
    with pytest.raises(GroundingError, match="inside"):
        validate_proposal(outside, width=180, height=100)


def test_normalized_model_coordinates_convert_to_original_pixels():
    normalized = json.loads(json.dumps(VALID))
    normalized["role_boxes"]["white_mug"] = [267, 692, 367, 901]
    normalized["role_points"]["white_handle"] = [300, 850]
    pixels = normalized_proposal_to_pixels(normalized, width=640, height=360)
    assert pixels["role_boxes"]["white_mug"] == [170, 249, 235, 325]
    assert pixels["role_points"]["white_handle"] == [192, 306]


def test_valid_reply_is_sha_bound_and_atomically_written(tmp_path):
    rgb = tmp_path / "INITIAL_RGB.png"
    output = tmp_path / "INITIAL_BOXES.json"
    frame(rgb)
    fake = client(tmp_path, [VALID])
    payload = ground_initial_frame(
        rgb,
        output,
        fake.key_file,
        maximum_calls=1,
        semantic_validator=lambda _proposal: diagnostics(),
        client=fake,
    )
    assert json.loads(output.read_text()) == payload
    assert payload["grounding"]["status"] == "valid"
    assert payload["grounding"]["actual_sam_depth_validated"] is True
    assert not list(tmp_path.glob(".INITIAL_BOXES.json.*"))


def test_actual_sam_depth_failure_gets_one_bounded_review(tmp_path):
    rgb = tmp_path / "INITIAL_RGB.png"
    output = tmp_path / "INITIAL_BOXES.json"
    frame(rgb)
    fake = client(tmp_path, [VALID, VALID])
    outcomes = iter([
        diagnostics(False, "table_patch has no actual SAM/depth support"),
        diagnostics(),
    ])
    payload = ground_initial_frame(
        rgb,
        output,
        fake.key_file,
        maximum_calls=2,
        semantic_validator=lambda _proposal: next(outcomes),
        client=fake,
    )
    assert payload["grounding"]["attempts"] == 2
    request2 = json.loads((tmp_path / "AUTONOMOUS_GROUNDING/REQUEST_02.json").read_text())
    assert "SAM/depth" in request2["validation_feedback"]["retry_reason"]
    assert fake.calls == 2


def test_semantic_diagnostics_are_complete_and_attributable():
    value = validate_semantic_diagnostics(diagnostics())
    assert value["role_pixel_count"]["white_handle"] == 20
    assert value["role_source"]["table_patch"] == "RGBD_ROI_plane"
    partial = diagnostics()
    partial["role_depth_count"].pop("table_patch")
    with pytest.raises(GroundingError, match="all requested roles"):
        validate_semantic_diagnostics(partial)


def test_offline_followup_can_start_from_bound_structured_feedback(tmp_path):
    rgb = tmp_path / "INITIAL_RGB.png"
    frame(rgb)
    fake = client(tmp_path, [VALID])
    prior = diagnostics(False, "table RGBD ROI does not support a 3mm plane")
    result = ground_initial_frame(
        rgb,
        tmp_path / "INITIAL_BOXES.json",
        fake.key_file,
        maximum_calls=1,
        semantic_validator=lambda _proposal: diagnostics(),
        initial_feedback=prior,
        client=fake,
    )
    assert result["grounding"]["attempts"] == 1
    request = json.loads((tmp_path / "AUTONOMOUS_GROUNDING/REQUEST_01.json").read_text())
    assert request["validation_feedback"]["role_source"]["table_patch"] == "RGBD_ROI_plane"


def test_missing_or_unknown_reply_fails_closed_without_output(tmp_path):
    rgb = tmp_path / "INITIAL_RGB.png"
    output = tmp_path / "INITIAL_BOXES.json"
    frame(rgb)
    unknown = {
        "status": "unknown",
        "role_boxes": {role: None for role in DEFAULT_CUP_ROLES},
        "role_points": {"white_handle": None},
        "reason": "handle is occluded",
    }
    fake = client(tmp_path, [unknown])
    with pytest.raises(GroundingUnknown):
        ground_initial_frame(rgb, output, fake.key_file, maximum_calls=1, client=fake)
    assert not output.exists()
    report = json.loads((tmp_path / "AUTONOMOUS_GROUNDING/UNKNOWN.json").read_text())
    assert report["status"] == "UNKNOWN" and report["output_created"] is False
    response = json.loads((tmp_path / "AUTONOMOUS_GROUNDING/RESPONSE_01.json").read_text())
    assert response["proposal"]["status"] == "unknown" and response["metrics"]["call_index"] == 1


def test_private_key_mode_is_enforced_and_never_enters_record(tmp_path):
    rgb = tmp_path / "INITIAL_RGB.png"
    frame(rgb)
    key = tmp_path / "key"
    key.write_text("secret-never-print", encoding="utf-8")
    key.chmod(0o644)
    fake = OpenRouterGrounder(key, maximum_calls=1, opener=lambda *_a, **_k: Response(VALID))
    with pytest.raises(GroundingUnknown):
        ground_initial_frame(rgb, tmp_path / "out.json", key, maximum_calls=1, client=fake)
    evidence = "".join(path.read_text() for path in (tmp_path / "AUTONOMOUS_GROUNDING").glob("*.json"))
    assert "secret-never-print" not in evidence


def test_existing_output_and_changed_frame_fail_closed(tmp_path):
    rgb = tmp_path / "INITIAL_RGB.png"
    output = tmp_path / "INITIAL_BOXES.json"
    frame(rgb)
    output.write_text("old", encoding="utf-8")
    fake = client(tmp_path, [VALID])
    with pytest.raises(GroundingError, match="fresh"):
        ground_initial_frame(rgb, output, fake.key_file, maximum_calls=1, client=fake)
    assert output.read_text() == "old"
