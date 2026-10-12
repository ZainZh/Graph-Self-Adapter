"""Actual read-only robot/FK/ZED/SAM preflight for V22 task roles."""
from pathlib import Path

import numpy as np

from prove.xtrainer.agent import write
from prove.xtrainer.io import NativeRPC, SafetyError, SafetyLayer
from prove.xtrainer.perception import ZEDCamera
from multitask_readonly_preflight import preflight
from real_feedback_stream import FramedFeedbackStream
from operator_launcher import startup_vision_class
from shared_cup_vision import SharedSAMVision


def run_readonly_shadow(*, task, runtime, required_roles, output, freeze):
    output = Path(output)
    if output.exists():
        raise SafetyError("fresh read-only shadow output required")
    output.mkdir(parents=True)
    write(output / "EXECUTION_FREEZE.json", freeze)
    streams = [FramedFeedbackStream(runtime["robot_ips"][arm], runtime["sdk_file"]) for arm in ("left", "right")]
    rpc = NativeRPC()
    vision_type = startup_vision_class(SharedSAMVision, SafetyError)
    vision = vision_type(
        ZEDCamera(serial=runtime["vision"].get("serial", 38968158)),
        np.asarray(runtime["T_left_from_camera"]),
        runtime["vision"],
        evidence_root=output / "SAM_MASK_EVIDENCE",
    )

    class ReadOnlyRobot:
        def samples(self):
            return [stream.snapshot() for stream in streams]
        def fk(self, *values):
            return rpc.fk(*values)
        def command(self, *values):
            raise SafetyError("read-only shadow cannot command")
        def home(self, *values):
            raise SafetyError("read-only shadow cannot home")
        def enable(self, *values):
            raise SafetyError("read-only shadow cannot enable")
        def force(self, *values):
            raise SafetyError("read-only shadow cannot apply force")

    try:
        for stream in streams:
            stream.connect()
        rpc.connect()
        vision.connect()
        result = preflight(
            ReadOnlyRobot(), vision, SafetyLayer(runtime["safety"]),
            runtime["known_initial_gripper_commands"], required_roles=required_roles,
        )
        if rpc.commands != 0:
            raise SafetyError("read-only shadow observed an unexpected command")
        result.update(task=task, mask_source="actual_SAM2", api_calls=0,
            command_interfaces_hard_rejected=True, device_connections=4)
        write(output / "RESULT.json", result)
        return result
    except Exception as error:
        write(output / "PREFLIGHT_FAILURE.json", {
            "task": task, "failure": str(error), "hardware_commands": rpc.commands,
            "api_calls": 0, "preflight_report": getattr(error, "preflight_report", None),
        })
        raise
    finally:
        rpc.close()
        vision.close()
        for stream in streams:
            stream.close()
