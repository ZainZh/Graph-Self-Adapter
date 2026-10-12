#!/usr/bin/env python3
"""Run actual SAM2 initialization and tracking on a retained RGB frame."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--utility", type=Path, required=True)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reply = json.loads(args.boxes.read_text())
    raw = args.image.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reply["frame_sha256"]:
        raise RuntimeError("boxes do not belong to the retained frame")
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    spec = importlib.util.spec_from_file_location("_server_sam_utility", args.utility)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    roles = list(reply["role_boxes"])
    tracker = module.make_sam2_tracker_factory(
        placeholders=roles, sam2_root=args.sam_root, checkpoint=args.checkpoint,
        config=args.config, device="cuda", autocast_dtype="bfloat16",
    )("server_validation")
    initialized = tracker.initialize(rgb, reply["role_boxes"])
    tracked = tracker.track(rgb.copy())
    report = {
        "schema": "graph-self-adapter.server-sam-forward-validation.v1",
        "image_sha256": reply["frame_sha256"],
        "roles": roles,
        "initialized_pixels": {role: int(np.asarray(initialized[role]).sum()) for role in roles},
        "tracked_pixels": {role: int(np.asarray(tracked[role]).sum()) for role in roles},
        "initialize_and_track_completed": True,
        "sam_backend": str((args.sam_root / "sam2/_C.py").resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
