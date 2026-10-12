#!/usr/bin/env python3
"""Wait for INITIAL_RGB.png and open the manual SAM prompt gate in-process."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

try:
    from tools.deployment.manual_sam_first_frame import (
        AnnotationCancelled,
        annotate,
        frame_sha256,
    )
except ModuleNotFoundError:  # Standalone execution from tools/deployment.
    from manual_sam_first_frame import AnnotationCancelled, annotate, frame_sha256


def wait_for_frame(
    frame_path: Path,
    *,
    timeout_s: float = 120.0,
    poll_s: float = 0.1,
    baseline_sha256: str | None = None,
) -> str:
    """Return SHA after two identical readable observations of a new PNG."""
    deadline = time.monotonic() + timeout_s
    previous: tuple[int, int, str] | None = None
    while time.monotonic() < deadline:
        try:
            stat = frame_path.stat()
            digest = frame_sha256(frame_path)
            current = (stat.st_size, stat.st_mtime_ns, digest)
        except (FileNotFoundError, PermissionError, OSError):
            previous = None
            time.sleep(poll_s)
            continue
        if stat.st_size > 0 and digest != baseline_sha256 and current == previous:
            return digest
        previous = current
        time.sleep(poll_s)
    raise TimeoutError(f"等待首帧超时：{frame_path}")


def wait_and_annotate(
    frame_path: Path,
    *,
    output_path: Path | None = None,
    timeout_s: float = 120.0,
    baseline_sha256: str | None = None,
) -> dict[str, object]:
    wait_for_frame(
        frame_path, timeout_s=timeout_s, baseline_sha256=baseline_sha256
    )
    return annotate(frame_path, output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--baseline-sha256")
    args = parser.parse_args(argv)
    try:
        payload = wait_and_annotate(
            args.frame,
            output_path=args.output,
            timeout_s=args.timeout,
            baseline_sha256=args.baseline_sha256,
        )
    except AnnotationCancelled as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"手动标注失败：{exc}", file=sys.stderr)
        return 1
    print(f"标注已确认并保存，frame_sha256={payload['frame_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
