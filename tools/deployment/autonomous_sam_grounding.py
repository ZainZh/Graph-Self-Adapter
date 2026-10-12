#!/usr/bin/env python3
"""Bounded Gemini proposal of first-frame SAM prompts; no camera or robot I/O.

The model proposes boxes and one foreground seed.  Local code validates exact
geometry and frame identity.  An optional caller-supplied semantic validator
must use actual SAM/depth evidence when the functional handle part is required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Callable, Mapping, Sequence
import urllib.error
import urllib.request
import uuid

from PIL import Image


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.8-flash"
IDENTITY = f"openrouter/{MODEL}"
REASONING = {"effort": "low", "exclude": True}
ROLE_ORDER = ("white_mug", "red_mug", "box", "white_handle", "table_patch")
DEFAULT_CUP_ROLES = {
    "white_mug": "the complete white mug object",
    "red_mug": "the complete red mug object",
    "box": "the wooden placement box target",
    "white_handle": "a small visible graspable outer part of the white mug with usable metric depth; avoid holes, background and occlusion",
    "table_patch": "a small unobstructed bare tabletop ROI next to the white mug for metric-depth plane fitting; this is not a SAM instance",
}
SYSTEM_PROMPT = """Locate only the requested roles in the supplied current RGB image. Return JSON only. All returned box and point coordinates use normalized 0..1000 image coordinates in [x0,y0,x1,y1] order, with x from left to right and y from top to bottom; local code converts them to the original image pixels. Boxes must be tight visible regions. white_mug and red_mug are whole SAM objects. box is the placement target SAM object. white_handle is a small visible solid graspable outer part of the white mug for SAM plus metric depth: avoid the mug's hollow interior, background seen through a handle, robot, glare and occluded edges. Its foreground seed must lie on visible solid material inside both white_handle and white_mug. table_patch is a bare tabletop ROI for metric-depth plane fitting, not a SAM instance; avoid all objects, robot and table edges. If any requested role or reliable solid white_handle seed/table ROI is not visible, return status unknown and no guessed geometry. These prompts do not authorize robot actions and do not claim successful segmentation, depth, grasp or task completion."""

ROLE_SOURCES = {
    "white_mug": "actual_SAM2",
    "red_mug": "actual_SAM2",
    "box": "actual_SAM2",
    "white_handle": "actual_SAM2_seed_depth_component",
    "table_patch": "RGBD_ROI_plane",
}


class GroundingError(RuntimeError):
    pass


class GroundingUnknown(GroundingError):
    pass


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_roles(roles: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(roles, Mapping) or tuple(roles) != ROLE_ORDER:
        raise GroundingError(f"roles must be ordered exactly as {ROLE_ORDER}")
    clean = {}
    for name in ROLE_ORDER:
        description = roles[name]
        if not isinstance(description, str) or not description.strip() or len(description) > 240:
            raise GroundingError(f"invalid short role semantics: {name}")
        clean[name] = description.strip()
    return clean


def validate_proposal(
    value: object, *, width: int, height: int
) -> tuple[dict[str, list[int]], dict[str, list[int]], str]:
    if not isinstance(value, dict) or set(value) != {"status", "role_boxes", "role_points", "reason"}:
        raise GroundingError("response must have exact status/role_boxes/role_points/reason keys")
    if value["status"] == "unknown":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
            raise GroundingError("unknown response requires a short reason")
        raise GroundingUnknown(reason.strip())
    if value["status"] != "proposal":
        raise GroundingError("status must be proposal or unknown")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        raise GroundingError("proposal requires a short reason")
    boxes = value["role_boxes"]
    points = value["role_points"]
    if not isinstance(boxes, dict) or set(boxes) != set(ROLE_ORDER):
        raise GroundingError("all and only five requested role boxes are required")
    if not isinstance(points, dict) or set(points) != {"white_handle"}:
        raise GroundingError("exactly one white_handle foreground seed is required")
    clean_boxes: dict[str, list[int]] = {}
    for role in ROLE_ORDER:
        raw = boxes[role]
        if not isinstance(raw, list) or len(raw) != 4 or any(not _finite_number(v) for v in raw):
            raise GroundingError(f"invalid box: {role}")
        if any(float(v) != int(v) for v in raw):
            raise GroundingError(f"box coordinates must be integers: {role}")
        box = [int(v) for v in raw]
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise GroundingError(f"box outside image or empty: {role}")
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise GroundingError(f"box too small: {role}")
        clean_boxes[role] = box
    raw_seed = points["white_handle"]
    if not isinstance(raw_seed, list) or len(raw_seed) != 2 or any(not _finite_number(v) for v in raw_seed):
        raise GroundingError("invalid white_handle foreground seed")
    if any(float(v) != int(v) for v in raw_seed):
        raise GroundingError("foreground seed coordinates must be integers")
    seed = [int(v) for v in raw_seed]
    if not (0 <= seed[0] < width and 0 <= seed[1] < height):
        raise GroundingError("foreground seed outside image")
    for role in ("white_handle", "white_mug"):
        x0, y0, x1, y1 = clean_boxes[role]
        if not (x0 <= seed[0] < x1 and y0 <= seed[1] < y1):
            raise GroundingError(f"foreground seed must lie inside {role}")
    return clean_boxes, {"white_handle": seed}, reason.strip()


def normalized_proposal_to_pixels(value: object, *, width: int, height: int) -> object:
    if not isinstance(value, dict) or value.get("status") != "proposal":
        return value
    boxes = value.get("role_boxes")
    points = value.get("role_points")
    if not isinstance(boxes, dict) or not isinstance(points, dict):
        return value
    converted = json.loads(json.dumps(value))
    for role, raw in boxes.items():
        if not isinstance(raw, list) or len(raw) != 4 or any(type(v) is not int or not 0 <= v <= 1000 for v in raw):
            raise GroundingError(f"invalid normalized box: {role}")
        x0, y0, x1, y1 = raw
        converted["role_boxes"][role] = [
            int(x0 * width // 1000),
            int(y0 * height // 1000),
            int(math.ceil(x1 * width / 1000)),
            int(math.ceil(y1 * height / 1000)),
        ]
    seed = points.get("white_handle")
    if not isinstance(seed, list) or len(seed) != 2 or any(type(v) is not int or not 0 <= v <= 1000 for v in seed):
        raise GroundingError("invalid normalized white_handle seed")
    converted["role_points"]["white_handle"] = [
        min(width - 1, int(round(seed[0] * width / 1000))),
        min(height - 1, int(round(seed[1] * height / 1000))),
    ]
    return converted


def validate_semantic_diagnostics(value: object) -> dict[str, object]:
    required = {
        "valid",
        "retry_reason",
        "role_pixel_count",
        "role_depth_count",
        "white_handle_seed_valid",
        "seed_in_roi",
        "segment_bbox",
        "role_source",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise GroundingError("semantic validator diagnostics are incomplete")
    if type(value["valid"]) is not bool or type(value["white_handle_seed_valid"]) is not bool or type(value["seed_in_roi"]) is not bool:
        raise GroundingError("semantic validator Boolean diagnostics are invalid")
    clean: dict[str, object] = {
        "valid": value["valid"],
        "white_handle_seed_valid": value["white_handle_seed_valid"],
        "seed_in_roi": value["seed_in_roi"],
    }
    for field in ("role_pixel_count", "role_depth_count"):
        counts = value[field]
        if not isinstance(counts, dict) or set(counts) != set(ROLE_ORDER):
            raise GroundingError(f"{field} must cover all requested roles")
        if any(type(counts[role]) is not int or counts[role] < 0 for role in ROLE_ORDER):
            raise GroundingError(f"{field} values must be nonnegative integers")
        clean[field] = {role: counts[role] for role in ROLE_ORDER}
    bboxes = value["segment_bbox"]
    if not isinstance(bboxes, dict) or set(bboxes) != set(ROLE_ORDER):
        raise GroundingError("segment_bbox must cover all requested roles")
    clean_bboxes = {}
    for role in ROLE_ORDER:
        box = bboxes[role]
        if box is not None and (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) is not int or v < 0 for v in box)
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise GroundingError(f"invalid segment_bbox: {role}")
        clean_bboxes[role] = box
    clean["segment_bbox"] = clean_bboxes
    sources = value["role_source"]
    if not isinstance(sources, dict) or sources != ROLE_SOURCES:
        raise GroundingError("role_source must distinguish SAM roles from RGBD ROI plane")
    clean["role_source"] = dict(sources)
    reason = value["retry_reason"]
    if value["valid"]:
        if reason is not None:
            raise GroundingError("valid semantic diagnostics cannot include retry_reason")
        if any(clean["role_pixel_count"][role] <= 0 or clean["role_depth_count"][role] <= 0 for role in ROLE_ORDER):
            raise GroundingError("valid semantic diagnostics require mask and depth for every role")
        if not clean["white_handle_seed_valid"] or not clean["seed_in_roi"]:
            raise GroundingError("valid semantic diagnostics require supported in-ROI handle seed")
    else:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
            raise GroundingError("invalid semantic diagnostics require a short retry_reason")
        reason = reason.strip()
    clean["retry_reason"] = reason
    return clean


def _response_schema() -> dict:
    box = {"anyOf": [
        {"type": "array", "prefixItems": [
            {"type": "integer", "minimum": 0, "maximum": 999},
            {"type": "integer", "minimum": 0, "maximum": 999},
            {"type": "integer", "minimum": 1, "maximum": 1000},
            {"type": "integer", "minimum": 1, "maximum": 1000},
        ], "minItems": 4, "maxItems": 4},
        {"type": "null"},
    ]}
    point = {"anyOf": [
        {"type": "array", "prefixItems": [
            {"type": "integer", "minimum": 0, "maximum": 1000},
            {"type": "integer", "minimum": 0, "maximum": 1000},
        ], "minItems": 2, "maxItems": 2},
        {"type": "null"},
    ]}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "initial_sam_grounding",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "role_boxes", "role_points", "reason"],
                "properties": {
                    "status": {"type": "string", "enum": ["proposal", "unknown"]},
                    "role_boxes": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(ROLE_ORDER),
                        "properties": {role: box for role in ROLE_ORDER},
                    },
                    "role_points": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["white_handle"],
                        "properties": {"white_handle": point},
                    },
                    "reason": {"type": "string", "maxLength": 240},
                },
            },
        },
    }


def _redact(value: object, secret: str = "") -> str:
    text = str(value)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:400]


def _read_private_key(path: Path) -> str:
    try:
        info = path.lstat()
    except OSError as error:
        raise GroundingError("API key file unavailable: " + _redact(error)) from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise GroundingError("API key must be a private regular file")
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise GroundingError("API key file must be owned by this user with mode 0600")
    secret = path.read_text(encoding="utf-8").strip()
    if not secret or len(secret) > 512 or any(character.isspace() for character in secret):
        raise GroundingError("invalid API key file")
    return secret


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


class OpenRouterGrounder:
    def __init__(
        self,
        key_file: Path,
        *,
        maximum_calls: int = 2,
        opener=urllib.request.urlopen,
        clock=time.monotonic,
    ):
        if type(maximum_calls) is not int or not 1 <= maximum_calls <= 2:
            raise GroundingError("initial grounding API budget must be one or two calls")
        self.key_file = Path(key_file)
        self.maximum_calls = maximum_calls
        self.opener = opener
        self.clock = clock
        self.calls = 0
        self.records: list[dict[str, object]] = []

    def ask(self, context: Mapping[str, object], image_bytes: bytes) -> tuple[dict, dict]:
        if self.calls >= self.maximum_calls:
            raise GroundingError("initial grounding API call budget exhausted")
        secret = _read_private_key(self.key_file)
        content = [
            {"type": "text", "text": json.dumps(context, separators=(",", ":"), allow_nan=False)},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")},
            },
        ]
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": 768,
            "temperature": 0,
            "reasoning": REASONING,
            "response_format": _response_schema(),
            "provider": {"require_parameters": True},
        }
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
            method="POST",
        )
        self.calls += 1
        started = self.clock()
        request_binding = {
            "model": MODEL,
            "endpoint": ENDPOINT,
            "context_sha256": sha256_bytes(json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
            "frame_sha256": sha256_bytes(image_bytes),
            "call_index": self.calls,
        }
        try:
            with self.opener(request, timeout=45) as response:
                raw = response.read(131073)
                if len(raw) > 131072:
                    raise GroundingError("API response size exceeded")
                body = json.loads(raw)
            content_value = body["choices"][0]["message"]["content"]
            value = json.loads(content_value) if isinstance(content_value, str) else content_value
            if not isinstance(value, dict):
                raise GroundingError("model response content must be a JSON object")
            usage = body.get("usage", {})
            safe_usage = {
                name: float(usage[name])
                for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
                if _finite_number(usage.get(name))
            }
            metrics = {
                **request_binding,
                "response_sha256": sha256_bytes(raw),
                "wall_sec": self.clock() - started,
                "usage": safe_usage,
                "identity": IDENTITY,
            }
            self.records.append(metrics)
            return value, metrics
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                detail = "HTTP " + str(error.code) + ": " + _redact(error.read(1024).decode(errors="replace"), secret)
            else:
                detail = "API/response error: " + _redact(error, secret)
            metrics = {**request_binding, "wall_sec": self.clock() - started, "identity": IDENTITY, "error": detail}
            self.records.append(metrics)
            raise GroundingError(detail) from None


def _load_frame(frame_path: Path) -> tuple[bytes, int, int]:
    raw = frame_path.read_bytes()
    with Image.open(frame_path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise GroundingError("current frame must be an RGB PNG")
        width, height = image.size
    if width < 16 or height < 16:
        raise GroundingError("current RGB frame is too small")
    return raw, width, height


def ground_initial_frame(
    frame_path: Path,
    output_path: Path,
    key_file: Path,
    *,
    roles: Mapping[str, str] = DEFAULT_CUP_ROLES,
    maximum_calls: int = 2,
    semantic_validator: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    evidence_root: Path | None = None,
    client: OpenRouterGrounder | None = None,
    initial_feedback: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a valid exact-frame prompt file or fail without creating it.

    ``semantic_validator`` is the shared-I/O hook.  For the cup task it must
    initialize actual SAM on this frame, intersect white_handle with the box,
    and run the existing seed/depth component and observed-patch gates.  It
    returns structured mask/depth/seed diagnostics with ``valid`` true or false.
    """
    frame_path = Path(frame_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise GroundingError("fresh grounding output path required")
    image_bytes, width, height = _load_frame(frame_path)
    frame_sha = sha256_bytes(image_bytes)
    role_semantics = validate_roles(roles)
    evidence_root = (Path(evidence_root).resolve() if evidence_root else output_path.parent / "AUTONOMOUS_GROUNDING")
    if client is None:
        client = OpenRouterGrounder(key_file, maximum_calls=maximum_calls)
    elif client.maximum_calls != maximum_calls:
        raise GroundingError("injected client call budget mismatch")
    if initial_feedback is not None:
        feedback = validate_semantic_diagnostics(initial_feedback)
        if feedback["valid"]:
            raise GroundingError("initial_feedback is only for a failed prior offline proposal")
    else:
        feedback = None
    failures: list[dict[str, object]] = []
    for attempt in range(1, maximum_calls + 1):
        raw_reply = None
        metrics = None
        diagnostics = None
        context = {
            "schema": "graph-self-adapter.autonomous-sam-grounding-request.v1",
            "task": "put_mug_on_box",
            "current_roles": role_semantics,
            "image": {"width": width, "height": height, "sha256": frame_sha},
            "attempt": attempt,
            "maximum_attempts": maximum_calls,
            "validation_feedback": feedback,
            "scope": "SAM prompt proposal only; no action authority",
            "coordinate_contract": "normalized_0_1000_xyxy; local deterministic conversion to original pixels",
        }
        request_record = {**context, "endpoint": ENDPOINT, "model": MODEL, "image_path_local_only": str(frame_path)}
        atomic_write_json(evidence_root / f"REQUEST_{attempt:02d}.json", request_record)
        try:
            raw_reply, metrics = client.ask(context, image_bytes)
            pixel_reply = normalized_proposal_to_pixels(raw_reply, width=width, height=height)
            boxes, points, reason = validate_proposal(pixel_reply, width=width, height=height)
            candidate = {
                "frame_sha256": frame_sha,
                "role_boxes": boxes,
                "role_points": points,
            }
            if sha256_file(frame_path) != frame_sha:
                raise GroundingError("current RGB frame changed during grounding")
            diagnostics = validate_semantic_diagnostics(semantic_validator(candidate)) if semantic_validator else None
            if diagnostics is not None and not diagnostics["valid"]:
                feedback = diagnostics
                raise GroundingUnknown(str(diagnostics["retry_reason"]))
            response_record = {
                "status": "accepted",
                "proposal": raw_reply,
                "pixel_proposal": pixel_reply,
                "proposal_sha256": sha256_bytes(json.dumps(raw_reply, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
                "metrics": metrics,
            }
            atomic_write_json(evidence_root / f"RESPONSE_{attempt:02d}.json", response_record)
            payload = {
                **candidate,
                "grounding": {
                    "schema": "graph-self-adapter.autonomous-sam-grounding.v1",
                    "status": "valid",
                    "source": IDENTITY,
                    "model": MODEL,
                    "attempts": attempt,
                    "request_sha256": sha256_file(evidence_root / f"REQUEST_{attempt:02d}.json"),
                    "response_sha256": sha256_file(evidence_root / f"RESPONSE_{attempt:02d}.json"),
                    "model_reason": reason,
                    "model_coordinate_space": "normalized_0_1000_xyxy",
                    "output_coordinate_space": "original_image_pixels_xyxy_exclusive",
                    "actual_sam_depth_validated": semantic_validator is not None,
                    "semantic_diagnostics": diagnostics,
                    "action_authority": False,
                },
            }
            if sha256_file(frame_path) != frame_sha:
                raise GroundingError("current RGB frame changed before atomic output")
            atomic_write_json(output_path, payload)
            return payload
        except GroundingUnknown as error:
            if not isinstance(feedback, dict):
                feedback = {"retry_reason": _redact(error)}
            failures.append({"attempt": attempt, "status": "unknown", "reason": _redact(error), "diagnostics": feedback})
            atomic_write_json(evidence_root / f"RESPONSE_{attempt:02d}.json", {
                **failures[-1], "proposal": raw_reply, "metrics": metrics,
                "proposal_sha256": (sha256_bytes(json.dumps(raw_reply, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) if raw_reply is not None else None),
            })
        except GroundingError as error:
            feedback = {"retry_reason": _redact(error)}
            failures.append({"attempt": attempt, "status": "invalid", "reason": _redact(error)})
            atomic_write_json(evidence_root / f"RESPONSE_{attempt:02d}.json", {
                **failures[-1], "proposal": raw_reply, "metrics": metrics,
                "proposal_sha256": (sha256_bytes(json.dumps(raw_reply, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) if raw_reply is not None else None),
            })
    atomic_write_json(
        evidence_root / "UNKNOWN.json",
        {
            "schema": "graph-self-adapter.autonomous-sam-grounding-unknown.v1",
            "status": "UNKNOWN",
            "frame_sha256": frame_sha,
            "attempts": failures,
            "output_created": False,
            "action_authority": False,
        },
    )
    raise GroundingUnknown("automatic grounding remained invalid/unknown after bounded review")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--maximum-calls", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        ground_initial_frame(
            args.frame,
            args.output,
            args.key_file,
            maximum_calls=args.maximum_calls,
            evidence_root=args.evidence_root,
        )
    except GroundingUnknown as error:
        print(f"UNKNOWN: {error}")
        return 2
    except GroundingError as error:
        print(f"FAILED_CLOSED: {error}")
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
