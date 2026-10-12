"""Cooperative scheduling derivative of frozen ``sim_robot_io``.

The original backend remains byte-for-byte unchanged.  This derivative yields
only after releasing the state lock when physics is behind wall-clock time, so
camera/feedback callers can acquire the same lock.  It never waits for a judge,
changes timestep, skips ``mj_step``, or changes physical timestamps.
"""
from __future__ import annotations

import time
import numpy as np

from prove.xtrainer.io import SafetyError
from sim_robot_io import SimRobot as FrozenSimRobot, SimRGBDCamera, SimStop


def fair_deadline_wait(stop_event, delay, *, sleeper=time.sleep):
    if delay > 0:
        stop_event.wait(delay)
    else:
        # Cooperative scheduler handoff after the state lock has been released.
        # sleep(0) does not encode physical time and does not pause on semantics.
        sleeper(0)


class FairSimRobot(FrozenSimRobot):
    scheduling_contract = "one cooperative yield after every behind-schedule unlocked physics tick"

    def _loop(self):
        from prove.task_graph.twin_sensor_adapter import raw_proprio_servo
        tick = float(self.world.model.opt.timestep)
        deadline = time.monotonic()
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if np.any(self.world.data.xfrc_applied) or np.any(self.world.data.qfrc_applied):
                        raise SafetyError("external force injection forbidden")
                    raw_proprio_servo(self.world, self.physics, self.q_target, self.g_target, np.zeros_like(self.q_target))
                    before = float(self.world.data.time)
                    self.mj.mj_step(self.world.model, self.world.data)
                    self.world._record_numerical_step(before)
                    self.world._update_bookkeeping()
                    self.physics_steps += 1
                    if not self.world._numerical_integrity_ok:
                        raise SafetyError("MuJoCo numerical integrity failure")
                    if self.physics_steps % 100 == 0:
                        self.world._sample_trace(self.q_target)
                deadline += tick
                delay = deadline - time.monotonic()
                fair_deadline_wait(self.stop_event, delay)
                if delay < -0.2:
                    deadline = time.monotonic()
        except Exception as exc:
            self.error = exc
            self.stop_event.set()


SimRobot = FairSimRobot

