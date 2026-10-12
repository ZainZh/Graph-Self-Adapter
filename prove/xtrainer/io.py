"""Explicit-connect Xtrainer I/O. Import/construct/check never connects hardware."""
from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
import socket
import threading
import time
import pickle
import re
import numpy as np
from scipy.spatial.transform import Rotation

class SafetyError(RuntimeError): pass

def array(value, shape):
    out=np.asarray(value,dtype=float)
    if out.shape!=shape or not np.isfinite(out).all(): raise SafetyError('invalid finite shape '+str(shape))
    return out

@dataclass
class RobotSample:
    stamp: float
    q: np.ndarray
    qdot: np.ndarray
    tcp: np.ndarray
    rotation: np.ndarray
    aperture: float|None
    source: str
    mode: int=5
    enabled: bool=True
    error: bool=False
    running: bool=False
    user: int=0
    tool: int=1
    user_offset: tuple=(0,0,0,0,0,0)
    tool_offset: tuple=(0,0,197,0,0,0)


def sdk_dtype(path):
    """Extract trusted hash-bound numpy dtype AST; do not import SDK drivers."""
    tree=ast.parse(Path(path).read_text())
    expr=next(n.value for n in tree.body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='MyType' for t in n.targets))
    allowed=(ast.Call,ast.Attribute,ast.Name,ast.Load,ast.Constant,ast.List,ast.Tuple,ast.Expression)
    if any(not isinstance(n,allowed) for n in ast.walk(ast.Expression(expr))):
        raise SafetyError('SDK dtype contains non-declarative expressions')
    if any(isinstance(n,ast.Name) and n.id!='np' for n in ast.walk(expr)):
        raise SafetyError('SDK dtype contains unexpected symbol')
    return eval(compile(ast.Expression(expr),str(path),'eval'),{'np':np,'__builtins__':{}})


def decode_feedback(raw,dtype,stamp):
    if len(raw)!=dtype.itemsize: raise SafetyError('feedback length mismatch')
    packet=np.frombuffer(raw,dtype=dtype,count=1)[0]
    if int(packet['len'])!=dtype.itemsize or int(packet['test_value'])!=0x123456789abcdef:
        raise SafetyError('feedback packet sentinel mismatch')
    pose=array(packet['tool_vector_actual'],(6,))
    sample=RobotSample(stamp,np.deg2rad(array(packet['q_actual'],(6,))),
        np.deg2rad(array(packet['qd_actual'],(6,))),pose[:3]*.001,
        Rotation.from_euler('xyz',pose[3:],degrees=True).as_matrix(),None,'dobot_TCP30004_actual',
        int(packet['robot_mode']),bool(packet['enable_status']),bool(packet['error_status']),
        bool(packet['running_status']),int(packet['user']),int(packet['tool']),tuple(packet['user[6]']),tuple(packet['tool[6]']))
    return int(packet['time_stamp']),sample


class FeedbackStream:
    def __init__(self,ip,sdk_file,*,clock=time.monotonic,socket_factory=socket.create_connection):
        self.ip=ip;self.dtype=sdk_dtype(sdk_file);self.clock=clock;self.socket_factory=socket_factory
        self.sock=None;self.thread=None;self.latest=None;self.error=None;self.stop_event=threading.Event()
    def connect(self):
        if self.sock: raise SafetyError('feedback already connected')
        self.sock=self.socket_factory((self.ip,30004),timeout=3);self.sock.settimeout(1)
        self.thread=threading.Thread(target=self._drain,daemon=True);self.thread.start()
    def _drain(self):
        try:
            previous=-1
            while not self.stop_event.is_set():
                raw=b''
                while len(raw)<self.dtype.itemsize:
                    chunk=self.sock.recv(self.dtype.itemsize-len(raw))
                    if not chunk: raise SafetyError('feedback disconnected')
                    raw+=chunk
                stamp,sample=decode_feedback(raw,self.dtype,self.clock())
                if stamp>previous: self.latest=sample;previous=stamp
        except Exception as exc:
            if not self.stop_event.is_set():self.error=exc
    def snapshot(self):
        if self.error:raise SafetyError('actual feedback failed: '+str(self.error))
        if self.latest is None or not 0<=self.clock()-self.latest.stamp<=.25:
            raise SafetyError('actual feedback missing/stale')
        return self.latest
    def close(self):
        self.stop_event.set()
        if self.sock:
            try:self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            self.sock.close()
        if self.thread:self.thread.join(timeout=2)


class NativeRPC:
    def __init__(self,*,endpoint='tcp://127.0.0.1:6001',timeout_ms=1000,socket_factory=None):
        if endpoint!='tcp://127.0.0.1:6001':raise SafetyError('only audited local service accepted')
        self.endpoint=endpoint;self.timeout_ms=timeout_ms;self.socket_factory=socket_factory
        self.sock=None;self.context=None;self.commands=0;self.connected=False
    def connect(self):
        if self.connected:raise SafetyError('already connected')
        if self.socket_factory:self.sock=self.socket_factory()
        else:
            import zmq
            self.context=zmq.Context();self.sock=self.context.socket(zmq.REQ)
            for option,value in ((zmq.RCVTIMEO,self.timeout_ms),(zmq.SNDTIMEO,self.timeout_ms),(zmq.LINGER,0)):
                self.sock.setsockopt(option,value)
        self.sock.connect(self.endpoint);self.connected=True
    def call(self,method,**args):
        if method not in ('get_fk','command_joint_state'):raise SafetyError('unaudited RPC forbidden')
        if not self.connected:raise SafetyError('RPC has not explicitly connected')
        try:
            self.sock.send(pickle.dumps({'method':method,'args':args}))
            result=pickle.loads(self.sock.recv())
        except Exception as exc:
            self.close();raise SafetyError('RPC timeout/transport failure: '+str(exc)) from exc
        if isinstance(result,dict) and (result.get('__robot_remote_error__') or result.get('error')):
            raise SafetyError('RPC remote error: '+str(result))
        return result
    @staticmethod
    def pack(q,g):
        q=array(q,(2,6));g=array(g,(2,))
        if np.any(g<0) or np.any(g>1):raise SafetyError('native aperture command must be 0closed..1open')
        return np.r_[q[0],g[0],q[1],g[1]]
    def command(self,q,g,flags):
        if flags not in ([True,False],[False,True],[True,True]):raise SafetyError('invalid native arm flags')
        result=self.call('command_joint_state',joint_state=self.pack(q,g),flag_in=flags)
        self.commands+=1;return result
    def fk(self,q,g,arm):
        result=array(self.call('get_fk',joint_state=self.pack(q,g)),(12,))
        index=0 if arm=='left' else 1;pose=result[index*6:index*6+6]
        return pose[:3],Rotation.from_rotvec(pose[3:]).as_matrix()
    def close(self):
        if self.sock:self.sock.close()
        if self.context:self.context.term()
        self.sock=None;self.connected=False


class DashboardStop:
    """Stop existing enabled robot only; no enable/home/configuration methods."""
    def __init__(self,ip,socket_factory=socket.create_connection):
        self.ip=ip;self.socket_factory=socket_factory;self.sock=None
    def connect(self):
        self.sock=self.socket_factory((self.ip,29999),timeout=2);self.sock.settimeout(2)
    def stop(self):
        if self.sock is None:raise SafetyError('stop connection absent')
        self.sock.sendall(b'ResetRobot()')
        raw=b''
        while b';' not in raw:
            chunk=self.sock.recv(4096)
            if not chunk or len(raw)>32768:raise SafetyError('ResetRobot response interrupted/oversized')
            raw+=chunk
        reply=raw.decode()
        match=re.fullmatch(r'\s*(-?\d+),\{(.*?)\},ResetRobot\(\);\s*',reply,flags=re.S)
        if not match or int(match[1])!=0:raise SafetyError('ResetRobot malformed/rejected: '+reply)
    def close(self):
        if self.sock:self.sock.close()


def load_calibration(left,right):
    import yaml
    rows=[yaml.safe_load(Path(p).read_text()) for p in (left,right)]
    transforms=[]
    for arm,row in zip(('left','right'),rows):
        if (row.get('arm')!=arm or row.get('camera_serial')!=38968158 or
            row.get('pose_xyz_unit')!='mm' or row.get('pose_euler_order')!='xyz'):
            raise SafetyError('calibration identity/unit mismatch')
        t=array(row['t_base_from_camera'],(4,4))
        if not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5):
            raise SafetyError('invalid extrinsic rigid transform')
        transforms.append(t)
    return transforms[0], transforms[0]@np.linalg.inv(transforms[1])


def dls_retarget(rpc,q,g,arm,delta_world,T_left_from_right,*,damping=.03,max_delta=.04):
    index=0 if arm=='left' else 1;delta=array(delta_world,(3,))
    native_delta=delta if index==0 else T_left_from_right[:3,:3].T@delta
    center,rotation=rpc.fk(q,g,arm);jac=np.zeros((6,6));eps=1e-4
    for j in range(6):
        probe=np.asarray(q).copy();probe[index,j]+=eps
        xyz,probe_rotation=rpc.fk(probe,g,arm);jac[:3,j]=(xyz-center)/eps
        jac[3:,j]=Rotation.from_matrix(probe_rotation@rotation.T).as_rotvec()/eps
    correction=jac.T@np.linalg.solve(jac@jac.T+damping**2*np.eye(6),np.r_[native_delta,np.zeros(3)])
    if not np.isfinite(correction).all() or np.max(np.abs(correction))>max_delta:
        raise SafetyError('DLS branch/step bound exceeded')
    result=np.asarray(q).copy();result[index]+=correction
    return result


class SafetyLayer:
    def __init__(self,config,clock=time.monotonic):self.config=config;self.clock=clock;self.last_target=None
    def samples(self,samples):
        for sample in samples:
            if not 0<=self.clock()-sample.stamp<=self.config['max_age_sec']:raise SafetyError('stale actual robot feedback')
            for value,shape in ((sample.q,(6,)),(sample.qdot,(6,)),(sample.tcp,(3,)),(sample.rotation,(3,3))):array(value,shape)
            if sample.error or not sample.enabled or sample.mode not in (5,7):raise SafetyError('robot mode/error not ready')
            if sample.user!=0 or sample.tool!=1 or not np.allclose(sample.user_offset,np.zeros(6),atol=1e-5) or not np.allclose(sample.tool_offset,[0,0,197,0,0,0],atol=.01):
                raise SafetyError('User0/Tool1 calibration mismatch')
            if np.max(np.abs(sample.qdot))>self.config['max_velocity_rad_sec']:raise SafetyError('actual velocity limit')
        q=np.stack([s.q for s in samples])
        if self.last_target is not None and np.max(np.abs(q-self.last_target))>self.config['max_tracking_rad']:
            raise SafetyError('command tracking error')
    def validate(self,q,g,samples,dt,rpc,T):
        self.samples(samples);q=array(q,(2,6));array(g,(2,));cfg=self.config
        current=np.stack([s.q for s in samples]);delta=q-current
        if dt<=0 or np.max(np.abs(delta))>cfg['max_step_rad'] or np.max(np.abs(delta))/dt>cfg['max_velocity_rad_sec']:
            raise SafetyError('joint step/velocity limit')
        if np.any(q<np.asarray(cfg['joint_min_rad'])) or np.any(q>np.asarray(cfg['joint_max_rad'])):
            raise SafetyError('joint position limit')
        for arm in ('left','right'):
            xyz,_=rpc.fk(q,g,arm)
            if arm=='right':xyz=(T@np.r_[xyz,1])[:3]
            if np.any(xyz<cfg['workspace_min_m']) or np.any(xyz>cfg['workspace_max_m']):raise SafetyError('TCP workspace limit')
        return True


def feedback_fk_check(rpc,samples,g,*,max_position_error_m=.01,max_rotation_error_rad=.03):
    """Compare independently measured native TCP against native controller FK."""
    q=np.stack([s.q for s in samples]);rows=[]
    for index,arm in enumerate(('left','right')):
        xyz,rotation=rpc.fk(q,g,arm)
        position=float(np.linalg.norm(xyz-samples[index].tcp))
        angle=float(np.linalg.norm(Rotation.from_matrix(rotation@samples[index].rotation.T).as_rotvec()))
        rows.append({'arm':arm,'position_error_m':position,'rotation_error_rad':angle,
                     'passed':position<=max_position_error_m and angle<=max_rotation_error_rad})
    return {'passed':all(row['passed'] for row in rows),'arms':rows,
            'frame':'each native robot base; calibrated User0/Tool1'}
