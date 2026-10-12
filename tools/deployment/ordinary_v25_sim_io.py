"""Real-capability simulator transport: metric aperture unavailable like real SDK."""
from dataclasses import replace
from kinematic_sim_robot_io import KinematicFairSimRobot,SimRGBDCamera,SimStop
class RealParitySimRobot(KinematicFairSimRobot):
    def samples(self):return [replace(sample,aperture=None) for sample in super().samples()]
SimRobot=RealParitySimRobot
