"""Production simulator I/O passes through the same public validators as real.
Current real SDK has no measured aperture; simulated finger encoders stay private.
"""
from dataclasses import replace
from sim_robot_io import SimRobot as BaseSimRobot, SimRGBDCamera as BaseCamera, SimStop as BaseStop
from three_task_v25_public_io import PublicRobot, PublicCamera, PublicStop

class _OwnTransport:
    def __init__(self,robot):self.robot=robot
    @property
    def commands(self):return self.robot.commands
    def samples(self):
        return [replace(sample,aperture=None,source='MuJoCo_own_robot_real_capability_proxy_aperture_unavailable')
                for sample in BaseSimRobot.samples(self.robot)]
    def command(self,*args):return BaseSimRobot.command(self.robot,*args)
    def fk(self,*args):return BaseSimRobot.fk(self.robot,*args)

class SimRobot(BaseSimRobot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.public_robot=PublicRobot(_OwnTransport(self),source='sim_real_capability')
    def _public(self):
        # Also supports zero-scene interface tests without instantiating physics.
        if not hasattr(self,'public_robot'):self.public_robot=PublicRobot(_OwnTransport(self),source='sim_real_capability')
        return self.public_robot
    def samples(self):return self._public().samples()
    def command(self,*args):return self._public().command(*args)
    def fk(self,*args):return self._public().fk(*args)

class SimRGBDCamera(BaseCamera):
    def connect(self):
        super().connect()
        self.public_camera=PublicCamera(_CameraTransport(self),self.T,source='sim_rendered_RGBD')
    def capture(self):return self.public_camera.capture()

class _CameraTransport:
    def __init__(self,camera):self.camera=camera
    def capture(self):return BaseCamera.capture(self.camera)

class SimStop(BaseStop):
    def stop(self):return PublicStop(_StopTransport(self)).stop()

class _StopTransport:
    def __init__(self,stop):self.stop_owner=stop
    def stop(self):return BaseStop.stop(self.stop_owner)
