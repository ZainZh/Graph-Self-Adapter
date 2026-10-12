"""Thin MuJoCo real-time I/O for the SAME object-centered hardware runtime.

Only actual own robot joints/FK and rendered RGB-D leave this backend. Object
state/contact bookkeeping is private evaluator work. RGB-D has no geom masks.
"""
import importlib.util
from pathlib import Path
import sys
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
from prove.xtrainer.io import RobotSample,SafetyError,array


def load_source(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


class SimRobot:
    def __init__(self,*,scene_run,physics_source,bundle,T_left_from_right,initial_q=None):
        import mujoco
        self.mj=mujoco;self.physics=load_source(physics_source,'_common_pipeline_twin_physics')
        self.world=self.physics.TwinWorld.from_run(scene_run,render=False,cup_release_control='portable_tcp_displacement_v1')
        self.T=array(T_left_from_right,(4,4));self.inverse=np.linalg.inv(self.T)
        self.lock=threading.RLock();self.fk_lock=threading.Lock();self.scratch=mujoco.MjData(self.world.model)
        self.q_target=np.asarray(bundle['initial_q'] if initial_q is None else initial_q).copy();self.g_target=bundle['initial_g'].copy()
        self.world.initialize(self.q_target,self.g_target);self.commands=0;self.physics_steps=0;self.error=None
        self.stop_event=threading.Event();self.thread=None;self._initialization_count=1
    def start(self):
        if self.thread is not None:raise SafetyError('simulation initialized/started twice')
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
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
                elif delay<-.2:deadline=time.monotonic() # scheduling only; never change physical time/state
        except Exception as exc:self.error=exc;self.stop_event.set()
    def samples(self):
        if self.error:raise SafetyError('simulation physics failed: '+str(self.error))
        with self.lock:
            p=self.world.proprioception();stamp=time.monotonic();travel=float(self.world.model.numeric('gripper_closing_travel').data[0])
        rows=[]
        for i in range(2):
            xyz=p.tcp_position[i];rotation=p.tcp_rotation[i]
            if i:xyz=(self.inverse@np.r_[xyz,1])[:3];rotation=self.inverse[:3,:3]@rotation
            moving=bool(np.max(abs(p.joint_velocity[i]))>.01)
            aperture=float(2*travel+p.finger_qpos[i,0]-p.finger_qpos[i,1])
            rows.append(RobotSample(stamp,p.joint_pos[i],p.joint_velocity[i],xyz,rotation,aperture,
                                    'MuJoCo_actual_robot_encoder_FK_proxy_not_hardware',mode=7 if moving else 5,running=moving))
        return rows
    def fk(self,q,g,arm):
        q=array(q,(2,6));g=array(g,(2,));index=0 if arm=='left' else 1
        with self.fk_lock:
            # Independent robot kinematics; do not copy hidden current object state.
            self.scratch.qpos[:]=self.world.model.qpos0
            self.scratch.qpos[self.world.arm_qpos_indices]=q.reshape(-1)
            travel=float(self.world.model.numeric('gripper_closing_travel').data[0])
            for joint,_,i,sign in self.world.fingers:self.scratch.qpos[joint.qposadr[0]]=sign*travel*(1-g[i])
            self.mj.mj_forward(self.world.model,self.scratch)
            site=self.scratch.site(arm+'_tcp');xyz=site.xpos.copy();rotation=site.xmat.reshape(3,3).copy()
        if index:return (self.inverse@np.r_[xyz,1])[:3],self.inverse[:3,:3]@rotation
        return xyz,rotation
    def command(self,q,g,flags):
        q=array(q,(2,6));g=array(g,(2,))
        with self.lock:
            for index,on in enumerate(flags):
                if on:self.q_target[index]=q[index];self.g_target[index]=g[index]
            self.commands+=1
    def hold(self,arm):
        index=0 if arm=='left' else 1
        with self.lock:self.q_target[index]=self.world.proprioception().joint_pos[index]
    def integrity(self):
        with self.lock:return {'initialization_count':self._initialization_count,'restore_count':self.world._state_writes_after_initialization,
            'external_force_injections':int(bool(np.any(self.world.data.xfrc_applied) or np.any(self.world.data.qfrc_applied))),
            'physics_steps':self.physics_steps,'physical_time_sec':float(self.world.data.time)}
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=3)
        self.world.close()


class SimStop:
    def __init__(self,robot,arm):self.robot=robot;self.arm=arm
    def stop(self):self.robot.hold(self.arm)


class SimRGBDCamera:
    def __init__(self,robot,*,width=640,height=360,camera='main'):
        self.robot=robot;self.width=width;self.height=height;self.camera=camera;self.renderer=None
        self.snapshot=None;self.K=None;self.T=None;self.last_physical_time=None
    def connect(self):
        self.renderer=self.robot.mj.Renderer(self.robot.world.model,height=self.height,width=self.width)
        self.snapshot=self.robot.mj.MjData(self.robot.world.model)
        with self.robot.lock:
            proxy=self.robot.physics.VisibleRGBD.__new__(self.robot.physics.VisibleRGBD)
            proxy.world=self.robot.world;proxy.width=self.width;proxy.height=self.height;proxy.camera=self.camera
            self.K,self.T=proxy._camera() # calibration only; NEVER _render/observe/segmentation
    def capture(self):
        if self.renderer is None:raise SafetyError('simulation RGBD camera not connected')
        stamp=time.monotonic()
        # Private graphics snapshot only; no object coordinates are returned to control.
        with self.robot.lock:
            data=self.robot.world.data;self.snapshot.qpos[:]=data.qpos;self.snapshot.qvel[:]=data.qvel
            self.snapshot.mocap_pos[:]=data.mocap_pos;self.snapshot.mocap_quat[:]=data.mocap_quat
            self.snapshot.time=data.time;self.last_physical_time=float(data.time)
        self.robot.mj.mj_forward(self.robot.world.model,self.snapshot)
        self.renderer.disable_segmentation_rendering();self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.snapshot,camera=self.camera);rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering();self.renderer.update_scene(self.snapshot,camera=self.camera)
        depth=self.renderer.render().copy();self.renderer.disable_depth_rendering()
        return stamp,rgb,depth,self.K.copy()
    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
