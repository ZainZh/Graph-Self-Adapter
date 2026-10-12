"""Real RGB-D/SAM roles; optional dependencies and devices are explicitly lazy."""
from dataclasses import dataclass,field
import hashlib
import importlib.util
from pathlib import Path
import time
import numpy as np
from .io import SafetyError,array

@dataclass
class Observation:
    stamp:float
    rgb:np.ndarray
    roles:dict
    source:str
    K:np.ndarray|None=None
    history:list=field(default_factory=list)


def role_geometry(depth,masks,K,T):
    depth=np.asarray(depth,dtype=float);K=array(K,(3,3));T=array(T,(4,4));out={}
    if depth.ndim!=2 or K[0,0]<=0 or K[1,1]<=0:raise SafetyError('invalid metric RGBD/intrinsics')
    for role,mask in masks.items():
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=depth.shape:raise SafetyError('mask/depth resolution mismatch')
        yy,xx=np.where(mask & np.isfinite(depth) & (depth>.05) & (depth<3.))
        if len(xx)<8:out[role]=None;continue
        z=depth[yy,xx];points=np.c_[(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z]
        shared=points@T[:3,:3].T+T[:3,3]
        out[role]=np.median(shared,axis=0)
    return out


class ZEDCamera:
    def __init__(self,*,serial=38968158,width=640,height=360,clock=time.monotonic):
        self.serial=serial;self.width=width;self.height=height;self.clock=clock;self.camera=None
    def connect(self):
        import pyzed.sl as sl
        self.sl=sl;self.camera=sl.Camera();params=sl.InitParameters()
        params.set_from_serial_number(self.serial);params.coordinate_units=sl.UNIT.METER
        params.depth_mode=sl.DEPTH_MODE.ULTRA
        if self.camera.open(params)!=sl.ERROR_CODE.SUCCESS:raise SafetyError('ZED open failed')
        resolution=sl.Resolution(self.width,self.height)
        self.resolution=resolution
        # SDK recalculates calibrated intrinsics for the actual requested image resolution.
        info=self.camera.get_camera_information(resolution)
        intrinsic=info.camera_configuration.calibration_parameters.left_cam
        self.K=np.array([[intrinsic.fx,0,intrinsic.cx],[0,intrinsic.fy,intrinsic.cy],[0,0,1.]])
        self.image=sl.Mat();self.depth=sl.Mat()
    def capture(self):
        if self.camera is None:raise SafetyError('camera not explicitly connected')
        stamp=self.clock();sl=self.sl
        if self.camera.grab()!=sl.ERROR_CODE.SUCCESS:raise SafetyError('ZED grab failed')
        self.camera.retrieve_image(self.image,sl.VIEW.LEFT,resolution=self.resolution)
        self.camera.retrieve_measure(self.depth,sl.MEASURE.DEPTH,resolution=self.resolution)
        bgra=self.image.get_data();rgb=np.asarray(bgra)[...,:3][...,::-1].copy()
        depth=np.asarray(self.depth.get_data(),dtype=float).copy()
        if rgb.shape[:2]!=depth.shape or rgb.shape[:2]!=(self.height,self.width):raise SafetyError('live resolution mismatch')
        return stamp,rgb,depth,self.K.copy()
    def close(self):
        if self.camera:self.camera.close()


class LiveVision:
    def __init__(self,camera,T,config,*,clock=time.monotonic):
        self.camera=camera;self.T=T;self.config=config;self.clock=clock;self.tracker=None
    def connect(self):self.camera.connect()
    def _tracker(self):
        path=Path(self.config['sam_utility']);spec=importlib.util.spec_from_file_location('_xtrainer_sam_utility',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        factory=module.make_sam2_tracker_factory(placeholders=list(self.config['role_boxes']),
            sam2_root=self.config['sam_root'],checkpoint=self.config['sam_checkpoint'],
            config=self.config['sam_config'],device=self.config.get('sam_device','cpu'),autocast_dtype=self.config.get('sam_autocast_dtype','bfloat16'))
        return factory('global')
    def capture(self):
        stamp,rgb,depth,K=self.camera.capture()
        if self.tracker is None:
            from PIL import Image
            import io
            buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
            sha=hashlib.sha256(buffer.getvalue()).hexdigest()
            if sha!=self.config.get('prompt_frame_sha256') or not self.config.get('role_boxes'):
                destination=Path(self.config['prompt_frame_output']);destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(buffer.getvalue())
                # Retain THIS first frame while the user labels it in a separate observation input file.
                import json
                prompt_file=Path(self.config['prompt_reply_path']);started=self.clock()
                while True:
                    if self.clock()-started>self.config.get('prompt_wall_timeout_sec',300):
                        raise SafetyError('first-frame prompt wall timeout; camera retained frame '+sha)
                    if prompt_file.exists():
                        reply=json.loads(prompt_file.read_text())
                        if reply.get('frame_sha256')==sha and reply.get('role_boxes'):
                            self.config['role_boxes']=reply['role_boxes'];break
                    time.sleep(.1)
            self.tracker=self._tracker();self.tracker.initialize(rgb,self.config['role_boxes'])
            # Initialization can be slow or human reviewed; resume on a NEW live frame and require freshness.
            stamp,rgb,depth,K=self.camera.capture();masks=self.tracker.track(rgb)
        else:masks=self.tracker.track(rgb)
        if not 0<=self.clock()-stamp<=self.config['max_age_sec']:raise SafetyError('RGBD/SAM processing too stale')
        roles=role_geometry(depth,masks,K,self.T)
        return Observation(stamp,rgb,roles,'ZED_metric_depth+SAM2_actual_prompt',K)
    def close(self):self.camera.close()
