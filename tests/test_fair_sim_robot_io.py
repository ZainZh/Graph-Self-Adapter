import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/deployment"))

from fair_sim_robot_io import FairSimRobot, FrozenSimRobot, fair_deadline_wait


class Stop:
    def __init__(self):
        self.waits = []
    def wait(self, value):
        self.waits.append(value)


def test_fair_backend_is_isolated_subclass_of_frozen_backend():
    assert issubclass(FairSimRobot, FrozenSimRobot)
    assert FairSimRobot is not FrozenSimRobot


def test_ahead_schedule_uses_interruptible_deadline_wait():
    stop = Stop()
    sleeps = []
    fair_deadline_wait(stop, 0.012, sleeper=sleeps.append)
    assert stop.waits == [0.012]
    assert sleeps == []


def test_behind_schedule_yields_only_after_unlocked_tick():
    stop = Stop()
    sleeps = []
    fair_deadline_wait(stop, -0.4, sleeper=sleeps.append)
    assert stop.waits == []
    assert sleeps == [0]
