"""Public I/O schema validation shared by both transports, without scene truth.

These adapters retain the existing camera tuple and robot methods. Metadata is
separate from policy data. Constructor and offline checks connect no devices.
"""
from dataclasses import dataclass
import time
import numpy as np
from prove.xtrainer.io import RobotSample, SafetyError, array

@dataclass(frozen=True)
class CommandAck:
    command_id: int
    accepted: bool
    stamp: float
    source: str
    physical_completion: bool = False

class PublicCamera:
    def __init__(self, transport, calibration, *, source):
        self.transport = transport; self.T = array(calibration, (4,4)); self.source = source
        self.frame_index = 0; self.metadata = None
    def connect(self): return self.transport.connect()
    def capture(self):
        stamp, rgb, depth, K = self.transport.capture()
        rgb = np.asarray(rgb); depth = np.asarray(depth)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise SafetyError('public RGBD shape/type mismatch')
        if not np.isfinite(stamp) or not np.isfinite(K).all(): raise SafetyError('public RGBD timestamp/intrinsics invalid')
        K = array(K, (3,3))
        if K[0,0] <= 0 or K[1,1] <= 0: raise SafetyError('public RGBD intrinsics invalid')
        self.frame_index += 1
        self.metadata = {'frame_id':f'{self.source}:{self.frame_index}', 'stamp_monotonic_sec':float(stamp),
                         'depth_units':'m', 'RGB_order':'RGB', 'K':K.tolist(), 'T_left_from_camera':self.T.tolist(),
                         'source':self.source, 'invalid_depth_preserved':True}
        return float(stamp), rgb, depth, K
    def close(self): return self.transport.close()

class PublicRobot:
    def __init__(self, transport, *, source):
        self.transport = transport; self.source = source; self.command_index = 0; self.last_ack = None
    @property
    def commands(self): return self.transport.commands
    def connect(self): return self.transport.connect()
    def samples(self):
        rows = self.transport.samples()
        if len(rows) != 2: raise SafetyError('public feedback requires two arm samples')
        for sample in rows:
            if not isinstance(sample, RobotSample) or not np.isfinite(sample.stamp): raise SafetyError('invalid public feedback')
            array(sample.q,(6,)); array(sample.qdot,(6,)); array(sample.tcp,(3,)); array(sample.rotation,(3,3))
            if sample.aperture is not None and (not np.isfinite(sample.aperture) or sample.aperture < 0):
                raise SafetyError('invalid metric actual aperture')
        return rows
    def command(self, q, g, flags):
        q = array(q,(2,6)); g = array(g,(2,))
        if flags not in ([True,False],[False,True],[True,True]) or np.any(g<0) or np.any(g>1):
            raise SafetyError('invalid public command')
        reply = self.transport.command(q,g,flags)
        if isinstance(reply,dict) and (reply.get('error') or reply.get('__robot_remote_error__')):
            raise SafetyError('public command rejected')
        self.command_index += 1
        self.last_ack = CommandAck(self.command_index, True, time.monotonic(), self.source)
        return reply
    def fk(self,q,g,arm):
        if arm not in ('left','right'): raise SafetyError('invalid public FK arm')
        position, rotation = self.transport.fk(array(q,(2,6)),array(g,(2,)),arm)
        return array(position,(3,)), array(rotation,(3,3))
    def close(self): return self.transport.close()

class PublicStop:
    def __init__(self,transport): self.transport=transport
    def stop(self): return self.transport.stop()

class OfflineMockTransport:
    """Explicit mock for wire/schema tests; no inference about real physics."""
    def __init__(self): self.commands=0; self.stopped=False; self.closed=False
    def connect(self): raise SafetyError('offline mock has no device connection')
    def capture(self): return time.monotonic(),np.zeros((4,6,3),np.uint8),np.ones((4,6)),np.array([[1.,0,2],[0,1,2],[0,0,1]])
    def samples(self):
        return [RobotSample(time.monotonic(),np.zeros(6),np.zeros(6),np.zeros(3),np.eye(3),None,'offline_mock') for _ in range(2)]
    def command(self,q,g,flags): self.commands+=1
    def fk(self,q,g,arm): return np.zeros(3),np.eye(3)
    def stop(self): self.stopped=True
    def close(self): self.closed=True

def offline_contract_check(task):
    observations=[]
    for backend in ('sim','real'):
        transport=OfflineMockTransport()
        camera=PublicCamera(transport,np.eye(4),source=backend+'_offline_mock')
        robot=PublicRobot(transport,source=backend+'_offline_mock'); stop=PublicStop(transport)
        stamp,rgb,depth,K=camera.capture(); samples=robot.samples()
        robot.command(np.zeros((2,6)),np.ones(2),[True,True]);robot.fk(np.zeros((2,6)),np.ones(2),'left');stop.stop()
        if robot.last_ack.physical_completion or not transport.stopped: raise SafetyError('contract mock failed')
        observations.append({'backend':backend,'mock_only':True,'camera_tuple_fields':['stamp','RGB','depth_m','K'],
                             'feedback_type':'RobotSample','arms':len(samples),'units':{'q':'rad','qdot':'rad/s','TCP':'m','aperture':'m'},
                             'methods':['capture','samples','command','fk','stop'], 'command_ack_is_not_completion':True,
                             'invalid_depth_and_unknown_aperture_preserved':True})
        camera.close();robot.close()
    return {'task':task,'mode':'offline_contract_check','device_connections':0,'hardware_commands':0,
            'shared_schema_and_units':True,'backend_checks':observations,
            'limits':'Offline mocks validate public interfaces only; real transport/device/physics equivalence untested'}

class PreparedRealTransport:
    """Real transport implementation prepared from host config; never auto-connects."""
    def __init__(self, config):
        from prove.xtrainer.io import NativeRPC, DashboardStop
        from real_feedback_stream import FramedFeedbackStream
        from prove.xtrainer.perception import ZEDCamera
        self.rpc=NativeRPC()
        self.streams=[FramedFeedbackStream(config['robot_ips'][arm],config['sdk_file']) for arm in ('left','right')]
        self.stops=[DashboardStop(config['robot_ips'][arm]) for arm in ('left','right')]
        self.camera=ZEDCamera(serial=config.get('vision',{}).get('serial',38968158))
    def connect(self):
        for stream in self.streams:stream.connect()
        for stop in self.stops:stop.connect()
        self.rpc.connect();self.camera.connect()
    def capture(self): return self.camera.capture()
    def stop(self):
        for stop in self.stops:stop.stop()
    @property
    def commands(self): return self.rpc.commands
    def samples(self): return [stream.snapshot() for stream in self.streams]
    def command(self,q,g,flags): return self.rpc.command(q,g,flags)
    def fk(self,q,g,arm): return self.rpc.fk(q,g,arm)
    def close(self):
        self.rpc.close();self.camera.close()
        for stream in self.streams:stream.close()
        for stop in self.stops:stop.close()

def real_constructor_check(config):
    """Any accidental connect path is intercepted and causes test failure."""
    from unittest.mock import patch
    from prove.xtrainer.io import FeedbackStream, NativeRPC, DashboardStop
    from prove.xtrainer.perception import ZEDCamera
    calls=[]
    def prohibited(*args,**kwargs):
        calls.append(True);raise SafetyError('device connection prohibited in offline constructor check')
    with patch('socket.create_connection',prohibited), patch.object(FeedbackStream,'connect',prohibited), patch.object(NativeRPC,'connect',prohibited), patch.object(DashboardStop,'connect',prohibited), patch.object(ZEDCamera,'connect',prohibited):
        transport=PreparedRealTransport(config)
        assert transport.rpc.sock is None and not transport.rpc.connected
        assert transport.camera.camera is None and all(stream.sock is None for stream in transport.streams)
        assert all(stop.sock is None for stop in transport.stops)
        transport.close()
    if calls: raise SafetyError('constructor tried connecting')
    return {'actual_real_classes_constructed':['NativeRPC','FramedFeedbackStream(left/right)','DashboardStop(left/right)','ZEDCamera'],
            'network_connect_calls':0,'camera_open_calls':0,'hardware_commands':0,
            'actual_aperture_capability':None,'motion_execution_enabled':False}

@dataclass
class BackendBundle:
    robot: PublicRobot
    camera: PublicCamera
    stops: list
    backend: str
    transport: object


def backend_bundle(backend, *, config, transport=None, calibration=None):
    """Both backends use the same production public wrappers.

    A simulator transport/camera must be provided by its scene factory. Real
    classes are lazy and require explicit connect; this factory never connects.
    """
    if backend not in ('sim','real'):raise SafetyError('unknown public backend')
    if backend=='real':
        transport=transport or PreparedRealTransport(config)
        camera_transport=transport.camera
    else:
        if transport is None or not hasattr(transport,'camera'):
            raise SafetyError('simulator transport with camera required')
        camera_transport=transport.camera
    T=calibration if calibration is not None else config.get('T_left_from_camera')
    if T is None:raise SafetyError('actual calibration required')
    robot=PublicRobot(transport,source=backend)
    camera=PublicCamera(camera_transport,T,source=backend)
    stop_transports=transport.stops
    return BackendBundle(robot,camera,[PublicStop(stop) for stop in stop_transports],backend,transport)
