"""P51 RGB/visible-mask planar reference tracker around unchanged P45/P47.

Four points are fixed material-plane reference points, NOT natural corners.
The current frozen G transfer FAILED offline 3D qualification. Learned wrappers
return diagnostic candidate_T_world_target but never an executable target.
A native development reference fixes the physical front-wall plane and metric
patch. Runtime accepts images, visible role masks, camera calibration and own
EEF pose only. Neither object state nor sites enter this interface.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from prove.fast_adapter.geometry3d_adapter import initial, pose, project
from prove.fast_adapter.p47_neural import P47Latent


def transform(p, R):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;return T


def visible_boxes(masks):
    boxes=np.zeros((3,4),np.float32);seen=np.zeros(3,np.float32)
    for i,role in enumerate(('actor','object','goal')):
        m=np.asarray(masks[role])
        if m.shape!=(448,448) or m.dtype!=bool:raise ValueError('Native448 boolean role masks required')
        y,x=np.nonzero(m)
        if len(x):boxes[i]=[x.min(),y.min(),x.max(),y.max()];seen[i]=1
    return boxes,seen


class P51PublicGrounder:
    def __init__(self, bundle):
        bundle=Path(bundle);self.reference=json.loads((bundle/'REFERENCE.json').read_text())
        self.rgb=cv2.cvtColor(cv2.imread(str(bundle/'reference_rgb.png')),cv2.COLOR_BGR2RGB)
        self.gray=cv2.cvtColor(self.rgb,cv2.COLOR_RGB2GRAY)
        self.face=np.load(bundle/'reference_masks.npz')['front_wall'].astype(bool)
        self.points=cv2.goodFeaturesToTrack(self.gray,120,.005,3,mask=self.face.astype(np.uint8)*255,blockSize=3)
        if self.points is None or len(self.points)<12:raise ValueError('Insufficient real reference-image features')
        self.patch=np.asarray(self.reference['reference_patch_pixels'],np.float64)
        self.T0=np.asarray(self.reference['T_world_reference'],float)
        self.offset=np.asarray(self.reference['T_reference_target'],float)

    def track(self, rgb, visible_caddy_mask):
        rgb=np.asarray(rgb);mask=np.asarray(visible_caddy_mask)
        if rgb.shape!=(448,448,3) or rgb.dtype!=np.uint8 or mask.shape!=(448,448) or mask.dtype!=bool:
            raise ValueError('Top-left native448 uint8 RGB and aligned visible bool mask required')
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        current,status,error=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,winSize=(15,15),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-7)
        info=dict(valid=False,reference_features=len(self.points),tracked_features=0,inliers=0,
                  method='reference_image_LK_forward_backward_and_RANSAC_homography',reason='tracking unavailable')
        if current is None:return None,info
        backward,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,current,None,winSize=(15,15),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-7)
        if backward is None:return None,info
        p=current[:,0];q=self.points[:,0]
        inside=(p[:,0]>=2)&(p[:,0]<446)&(p[:,1]>=2)&(p[:,1]<446)
        ij=np.rint(np.clip(p,0,447)).astype(int)
        keep=status[:,0].astype(bool)&back_status[:,0].astype(bool)&inside&mask[ij[:,1],ij[:,0]]
        keep &= (np.linalg.norm(backward[:,0]-q,axis=1)<1.)&(error[:,0]<20.)
        info['tracked_features']=int(keep.sum())
        if keep.sum()<12:return None,dict(info,reason='fewer than twelve visible image-supported tracks')
        H,inlier=cv2.findHomography(q[keep],p[keep],cv2.RANSAC,1.5)
        if H is None:return None,dict(info,reason='homography unavailable')
        inlier=inlier[:,0].astype(bool);info['inliers']=int(inlier.sum())
        if inlier.sum()<12 or inlier.mean()<.7:return None,dict(info,reason='insufficient homography consensus')
        corners=cv2.perspectiveTransform(self.patch.astype(np.float32)[None],H)[0]
        hull=cv2.convexHull(p[keep][inlier].astype(np.float32))
        if any(cv2.pointPolygonTest(hull,tuple(map(float,x)),False)<0 for x in corners):
            return None,dict(info,reason='reference points lack surrounding visible image support')
        if not cv2.isContourConvex(corners) or cv2.contourArea(corners,oriented=True)*cv2.contourArea(self.patch.astype(np.float32),oriented=True)<=0:
            return None,dict(info,reason='reflected or nonconvex reference point order')
        for x,y in np.rint(corners).astype(int):
            if x<2 or x>=446 or y<2 or y>=446 or mask[y-2:y+3,x-2:x+3].mean()<.8:
                return None,dict(info,reason='reference surface point occluded or outside visible mask')
        fitted=cv2.perspectiveTransform(q[keep][inlier][None],H)[0]
        info.update(valid=True,reason='image-supported visible planar reference',homography=H.tolist(),
                    reprojection_rms_px=float(np.sqrt(np.mean(np.sum((fitted-p[keep][inlier])**2,axis=1)))))
        return corners.astype(np.float32),info

    def observation(self, *, rgb, role_masks, camera_position, camera_rotation, K, position, rotation):
        corners,quality=self.track(rgb,role_masks['goal'])
        if corners is None:return None,quality
        B=self.T0[:3,:3];origin=self.T0[:3,3]
        boxes,seen=visible_boxes(role_masks)
        # A single immutable teacher coordinate change makes nominal patch R=I.
        obs=dict(corners=corners,K=np.asarray(K,float),camera_position=B.T@(np.asarray(camera_position)-origin),
                 camera_rotation=B.T@np.asarray(camera_rotation),position=B.T@(np.asarray(position)-origin),
                 rotation=B.T@np.asarray(rotation),role_bbox=boxes,role_visible=seen,local=np.zeros(48,np.float32))
        from prove.task_graph.fast_adapter_bridge import public_observation
        obs=public_observation(obs,'geometry')
        initial_points=initial(obs,'insert');p,R,size=pose(initial_points)
        if R[2,2]<.5 or np.dot(obs['camera_position']-p,R[:,2])<=0:
            return None,dict(quality,valid=False,reason='front normal or cyclic point order inconsistent with fixed reference')
        quality.update(nominal_pnp_reference=self.T0@transform(p,R),nominal_half_size=size.tolist(),
                       fixed_camera_single_center=True,multiview_identifiability_claim=False)
        return obs,quality


class P51FrozenAdapterGrounder(P51PublicGrounder):
    def __init__(self,bundle,model,arm='continuous'):
        super().__init__(bundle)
        if model.task!='insert' or model.representation!='geometry':raise ValueError('Frozen insert/G head required')
        self.adapter=P47Latent(model,arm=arm,device='cpu')

    def step(self,t,**public):
        obs,quality=self.observation(**public)
        if obs is None:return dict(valid=False,quality=quality,T_world_target=None,observation=None)
        self.adapter.update(obs,t)
        estimate=self.adapter.estimate()
        if estimate is None:return dict(valid=False,quality=quality,T_world_target=None,observation=obs)
        p,R,size=estimate
        T=self.T0@transform(p,R)
        quality['corrected_reprojection_rms_px']=float(np.sqrt(np.mean(np.sum((project(self.adapter.points,obs)-obs['corners'])**2,axis=1))))
        return dict(valid=True,execution_qualified=False,qualification_failure='frozen_G_LIBERO_3D_transfer_failed',
                    quality=quality,observation=obs,T_world_reference=T,T_world_target=None,candidate_T_world_target=T@self.offset,
                    nominal_T_world_target=quality['nominal_pnp_reference']@self.offset,estimated_half_size=size,
                    adaptation_record=self.adapter.records[-1])


class P51MultiviewFrozenAdapterGrounder:
    """Two declared calibrated RGB channels feed ONE unchanged persistent z."""
    def __init__(self,bundle,model,arm='continuous'):
        self.channels={name:P51PublicGrounder(Path(bundle)/name) for name in ('agentview','frontview')}
        first=self.channels['agentview'];self.T0=first.T0;self.offset=first.offset
        if any(not np.array_equal(g.T0,self.T0) or not np.array_equal(g.offset,self.offset) for g in self.channels.values()):
            raise ValueError('Both image templates must share one physical reference/target relation')
        if model.task!='insert' or model.representation!='geometry':raise ValueError('Frozen insert/G head required')
        self.adapter=P47Latent(model,arm=arm,device='cpu');self.last_t=-1;self.seen_channels=set()

    def step(self,t,*,channel,**public):
        if channel not in self.channels or t<=self.last_t:raise ValueError('Declared channel and increasing observation index required')
        self.last_t=t
        obs,quality=self.channels[channel].observation(**public)
        quality.update(channel=channel,declared_channels=list(self.channels))
        if obs is None:return dict(valid=False,quality=quality,T_world_target=None,observation=None)
        self.seen_channels.add(channel);self.adapter.update(obs,t)
        p,R,size=self.adapter.estimate();T=self.T0@transform(p,R)
        if not np.isfinite(T).all():return dict(valid=False,quality=dict(quality,reason='nonfinite frozen-head estimate'),T_world_target=None,observation=obs)
        quality.update(fixed_camera_single_center=len(self.seen_channels)==1,observed_channels=sorted(self.seen_channels),
                       corrected_reprojection_rms_px=float(np.sqrt(np.mean(np.sum((project(self.adapter.points,obs)-obs['corners'])**2,axis=1)))))
        return dict(valid=True,execution_qualified=False,qualification_failure='frozen_G_LIBERO_3D_transfer_failed',
                    quality=quality,observation=obs,T_world_reference=T,T_world_target=None,candidate_T_world_target=T@self.offset,
                    nominal_T_world_target=quality['nominal_pnp_reference']@self.offset,estimated_half_size=size,
                    adaptation_record=self.adapter.records[-1])
