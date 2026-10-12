"""Same simulator physics with cooperative ticks and bounded public state reads.
Own sample/camera timestamps retain their original acquisition time; timeout is
an explicit error. Private state is copied only inside the graphics backend.
"""
from contextlib import contextmanager
from dataclasses import replace
import time
import numpy as np
from prove.xtrainer.io import SafetyError
from sim_robot_io import SimRobot as FrozenSimRobot, SimRGBDCamera as FrozenCamera, SimStop as FrozenStop
from three_task_v25_public_io import PublicRobot,PublicCamera,PublicStop

PUBLIC_LOCK_TIMEOUT_SEC=.25

@contextmanager
def public_state_lock(robot,operation):
    if not robot.lock.acquire(timeout=PUBLIC_LOCK_TIMEOUT_SEC):
        raise SafetyError('simulator public '+operation+' state lock timeout (.25s)')
    try:yield
    finally:robot.lock.release()

class _OwnTransport:
    def __init__(self,robot):self.robot=robot
    @property
    def commands(self):return self.robot.commands
    def samples(self):
        with public_state_lock(self.robot,'samples'):
            samples=FrozenSimRobot.samples(self.robot)
        return [replace(row,aperture=None,source='MuJoCo_own_robot_real_capability_bounded_feedback') for row in samples]
    def command(self,*args):
        with public_state_lock(self.robot,'command'):return FrozenSimRobot.command(self.robot,*args)
    def fk(self,*args):return FrozenSimRobot.fk(self.robot,*args)

class SimRobot(FrozenSimRobot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.public_robot=PublicRobot(_OwnTransport(self),source='sim_real_capability')
    def _public(self):
        if not hasattr(self,'public_robot'):self.public_robot=PublicRobot(_OwnTransport(self),source='sim_real_capability')
        return self.public_robot
    def samples(self):return self._public().samples()
    def command(self,*args):return self._public().command(*args)
    def fk(self,*args):return self._public().fk(*args)
    def hold(self,arm):
        with public_state_lock(self,'stop'):return FrozenSimRobot.hold(self,arm)
    def _loop(self):
        from prove.task_graph.twin_sensor_adapter import raw_proprio_servo
        tick=float(self.world.model.opt.timestep);deadline=time.monotonic()
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if np.any(self.world.data.xfrc_applied) or np.any(self.world.data.qfrc_applied):raise SafetyError('external force injection forbidden')
                    raw_proprio_servo(self.world,self.physics,self.q_target,self.g_target,np.zeros_like(self.q_target))
                    before=float(self.world.data.time);self.mj.mj_step(self.world.model,self.world.data)
                    self.world._record_numerical_step(before);self.world._update_bookkeeping();self.physics_steps+=1
                    if not self.world._numerical_integrity_ok:raise SafetyError('MuJoCo numerical integrity failure')
                    if self.physics_steps%100==0:self.world._sample_trace(self.q_target)
                deadline+=tick;delay=deadline-time.monotonic()
                if delay>0:self.stop_event.wait(delay)
                else:time.sleep(0)  # yield after releasing state lock; no physical time/state change
                if delay<-.2:deadline=time.monotonic()
        except Exception as error:self.error=error;self.stop_event.set()

class SimRGBDCamera(FrozenCamera):
    def connect(self):
        super().connect();self.public_camera=PublicCamera(_RawCamera(self),self.T,source='sim_rendered_RGBD_bounded_state')
    def capture(self):return self.public_camera.capture()
    def raw_capture(self):
        if self.renderer is None:raise SafetyError('simulation RGBD camera not connected')
        stamp=time.monotonic()  # retained even when acquiring the lock takes time
        with public_state_lock(self.robot,'camera'):
            data=self.robot.world.data;self.snapshot.qpos[:]=data.qpos;self.snapshot.qvel[:]=data.qvel
            self.snapshot.mocap_pos[:]=data.mocap_pos;self.snapshot.mocap_quat[:]=data.mocap_quat
            self.snapshot.time=data.time;self.last_physical_time=float(data.time)
        self.robot.mj.mj_forward(self.robot.world.model,self.snapshot)
        self.renderer.disable_segmentation_rendering();self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.snapshot,camera=self.camera);rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering();self.renderer.update_scene(self.snapshot,camera=self.camera)
        depth=self.renderer.render().copy();self.renderer.disable_depth_rendering()
        return stamp,rgb,depth,self.K.copy()

class _RawCamera:
    def __init__(self,camera):self.camera=camera
    def capture(self):return self.camera.raw_capture()

class SimStop(FrozenStop):
    def stop(self):return PublicStop(_RawStop(self)).stop()
class _RawStop:
    def __init__(self,stop):self.stop_owner=stop
    def stop(self):return FrozenStop.stop(self.stop_owner)
