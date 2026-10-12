from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/deployment"))

from fair_sim_robot_io import FairSimRobot
from kinematic_sim_robot_io import KinematicFairSimRobot


def test_candidate_is_isolated_fair_subclass():
    assert issubclass(KinematicFairSimRobot, FairSimRobot)
    assert KinematicFairSimRobot.fk is not FairSimRobot.fk
