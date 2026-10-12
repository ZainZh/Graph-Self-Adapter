from pathlib import Path
import sys,time,threading
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_v25_fair_public_sim_io import SimRobot,SimRGBDCamera,PUBLIC_LOCK_TIMEOUT_SEC
from prove.xtrainer.io import SafetyError

def fake_robot():
    robot=SimRobot.__new__(SimRobot);robot.lock=threading.RLock();robot.stop_event=threading.Event();robot.thread=None
    robot.error=None;robot.commands=0;robot.physics_steps=0;robot.q_target=np.zeros((2,6));robot.g_target=np.ones(2);robot.inverse=np.eye(4);robot.physics=None
    model=SimpleNamespace(opt=SimpleNamespace(timestep=.001),numeric=lambda name:SimpleNamespace(data=[.036]))
    data=SimpleNamespace(time=0.,xfrc_applied=np.zeros(1),qfrc_applied=np.zeros(1))
    proprio=lambda:SimpleNamespace(tcp_position=np.zeros((2,3)),tcp_rotation=np.repeat(np.eye(3)[None],2,axis=0),joint_velocity=np.zeros((2,6)),joint_pos=np.zeros((2,6)),finger_qpos=np.zeros((2,2)))
    robot.world=SimpleNamespace(model=model,data=data,proprioception=proprio,_numerical_integrity_ok=True,_record_numerical_step=lambda before:None,_update_bookkeeping=lambda:None,_sample_trace=lambda q:None)
    def step(model,data):time.sleep(.003);data.time+=model.opt.timestep
    robot.mj=SimpleNamespace(mj_step=step)
    return robot

def test_overloaded_ticks_allow_bounded_public_samples_without_step_skips(monkeypatch):
    import prove.task_graph.twin_sensor_adapter as servo
    monkeypatch.setattr(servo,'raw_proprio_servo',lambda *args:None)
    robot=fake_robot();worker=threading.Thread(target=robot._loop);worker.start()
    durations=[]
    try:
        for _ in range(30):
            start=time.monotonic();rows=robot.samples();durations.append(time.monotonic()-start)
            assert all(row.aperture is None for row in rows)
            time.sleep(.001)
    finally:robot.stop_event.set();worker.join(1)
    assert not worker.is_alive() and robot.error is None
    assert max(durations)<PUBLIC_LOCK_TIMEOUT_SEC
    assert robot.physics_steps>0 and robot.world.data.time==pytest.approx(robot.physics_steps*.001)

@pytest.mark.parametrize('operation',['samples','camera'])
def test_public_lock_timeout_is_explicit_and_bounded(operation):
    robot=fake_robot();held=threading.Event();release=threading.Event()
    def hog():
        with robot.lock:held.set();release.wait(1)
    worker=threading.Thread(target=hog);worker.start();assert held.wait(1)
    start=time.monotonic()
    try:
        with pytest.raises(SafetyError,match='state lock timeout'):
            if operation=='samples':robot.samples()
            else:
                camera=SimRGBDCamera.__new__(SimRGBDCamera);camera.renderer=object();camera.robot=robot;camera.raw_capture()
        assert PUBLIC_LOCK_TIMEOUT_SEC<=time.monotonic()-start<.40
    finally:release.set();worker.join(1)
