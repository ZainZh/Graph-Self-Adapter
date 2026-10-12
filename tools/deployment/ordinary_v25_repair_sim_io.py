"""Simulator RGBD uses exact common public validator and unavailable real aperture."""
from ordinary_v25_sim_io import RealParitySimRobot
from kinematic_sim_robot_io import SimRGBDCamera as BaseCamera,SimStop as BaseStop
from three_task_v25_public_io import PublicCamera,PublicStop
SimRobot=RealParitySimRobot
class RawCapture:
    def __init__(self,owner):self.owner=owner
    def capture(self):return BaseCamera.capture(self.owner)
class SimRGBDCamera(BaseCamera):
    def connect(self):
        super().connect();self.public=PublicCamera(RawCapture(self),self.T,source='sim_real_public_contract')
    def capture(self):
        self.latest_public_capture=self.public.capture();return self.latest_public_capture
class SimStop:
    def __init__(self,*args):self.public=PublicStop(BaseStop(*args))
    def stop(self):return self.public.stop()
