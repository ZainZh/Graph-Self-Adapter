"""Geometry-only online latent diagnostic: calibrated images -> SE(3).

This is gradient geometry estimation, not a trained DINO adapter. Closed-form
multi-view estimation is retained as a strong same-information reference.
"""
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation

def rays(obs):
    uv=obs['corners'];K=obs['K'];v=np.c_[(uv[:,0]-K[0,2])/K[0,0],-(uv[:,1]-K[1,2])/K[1,1],-np.ones(4)]
    v=v@obs['camera_rotation'].T;return v/np.linalg.norm(v,axis=1,keepdims=True)

def project(points,obs):
    cam=(points-obs['camera_position'])@obs['camera_rotation'];K=obs['K']
    return np.c_[K[0,2]+K[0,0]*cam[:,0]/(-cam[:,2]),K[1,2]-K[1,1]*cam[:,1]/(-cam[:,2])]

def initial(obs, task):
    # Training nominal functional-frame dimensions, not test object geometry.
    a,b=(.019,.016) if task=='insert' else (.018,.019)
    points=np.array([[-a,b,0],[a,b,0],[a,-b,0],[-a,-b,0]],np.float64)
    options=[]
    for shift in range(4):
        uv=np.roll(obs['corners'],-shift,axis=0).astype(np.float64)
        ok,rv,tv=cv2.solvePnP(points,uv,obs['K'],None,flags=cv2.SOLVEPNP_IPPE)
        if not ok:continue
        cvR=cv2.Rodrigues(rv)[0];gl=np.diag([1,-1,-1]);R=obs['camera_rotation']@gl@cvR
        center=obs['camera_position']+obs['camera_rotation']@gl@tv[:,0]
        # Resolve an unmarked rectangle's cyclic symmetry using the declared
        # nominal skill orientation; no test target attitude is provided.
        nominal=np.eye(3) if task=='insert' else Rotation.from_euler('x',90,degrees=True).as_matrix()
        score=np.linalg.norm(Rotation.from_matrix(R@nominal.T).as_rotvec())
        options.append((score,points@R.T+center))
    if not options:raise RuntimeError('Initial planar pose failed')
    return min(options,key=lambda x:x[0])[1]

def pose(points):
    center=points.mean(0);x=points[1]+points[2]-points[0]-points[3];x=x/np.linalg.norm(x)
    y=points[0]+points[1]-points[2]-points[3];y=y-x*np.dot(x,y);y=y/np.linalg.norm(y)
    z=np.cross(x,y);R=np.stack([x,y,z],axis=1)
    size=np.array([np.mean([np.linalg.norm(points[1]-points[0]),np.linalg.norm(points[2]-points[3])]),np.mean([np.linalg.norm(points[0]-points[3]),np.linalg.norm(points[1]-points[2])])])/2
    return center,R,size

class GeometryLatent:
    def __init__(self,task,arm):self.task=task;self.arm=arm;self.points=None;self.history=[];self.losses=[]
    def update(self,obs,t):
        if obs['corners'] is None:return
        if self.points is None:self.points=initial(obs,self.task)
        if self.arm=='single':self.points=initial(obs,self.task);return
        if self.arm=='fixed' or (self.arm=='prefix' and t>=12):return
        uv=project(self.points,obs);options=[np.roll(obs['corners'],-k,axis=0) for k in range(4)]
        aligned=min(options,key=lambda x:np.square(x-uv).sum())
        current=dict(obs,corners=aligned);v=rays(current);eye=np.eye(3);A=eye[None]-v[:,:,None]*v[:,None,:];C=obs['camera_position']
        self.history.append((A,C))
        # Preserve the common parallax prefix and a recent window. Once the tool
        # is stationary, discarding every earlier view makes depth unobservable.
        history=self.history if len(self.history)<=32 else self.history[:12]+self.history[-20:]
        H=np.mean([a for a,c in history],axis=0)
        b=np.mean([np.einsum('kij,j->ki',a,c) for a,c in history],axis=0)
        if len(self.history)<3:return
        if self.arm=='multiview':
            estimate=np.linalg.solve(H+eye[None]*1e-7,b[:,:,None])[:,:,0]
            if np.isfinite(estimate).all():self.points=estimate
            return
        # Actual autograd update of a persistent 12-D geometry latent. The
        # damped geometry Hessian preconditioner is analytic and not meta-learned.
        z=torch.tensor(self.points,dtype=torch.float64,requires_grad=True)
        h=torch.tensor(H);rhs=torch.tensor(b);pre=torch.linalg.inv(h+torch.eye(3)[None]*1e-5)
        for _ in range(3):
            loss=(.5*torch.einsum('ki,kij,kj->k',z,h,z)-(rhs*z).sum(-1)).sum()
            grad,=torch.autograd.grad(loss,z)
            dz=torch.einsum('kij,kj->ki',pre,grad)
            dz=dz*torch.clamp(.012/dz.norm(dim=-1).clamp_min(1e-12),max=1)[:,None]
            z=(z-.65*dz).detach().requires_grad_(True)
        self.points=z.detach().numpy();self.losses.append(float(loss.detach()))

    def estimate(self):return None if self.points is None else pose(self.points)

def command(task,t,estimate,start_p,start_R,length=0.,anchor=None):
    if t<12:
        # Identical short camera/tool moves for every arm; no test target access.
        waypoints=np.array([[.018,0,0],[0,.018,0],[-.018,0,0],[0,-.018,.01]])
        return start_p+start_R@waypoints[t//3],start_R,False,'共同小幅观察'
    if estimate is None:return start_p,start_R,False,'目标不可见，保持'
    p,R,size=estimate
    if task=='insert':
        # The mask frame is the outer rim; dimensions affect orientation/center,
        # while the target relative depth is shared across all fitting objects.
        clearance=.04 if t<60 else (.004 if t<90 else -.04)
        return p+R[:,2]*(length+clearance),R,False,('三维位置与朝向对齐' if t<60 else '靠近槽口' if t<90 else '沿估计槽轴插入')
    grip=p+R[:,0]*(size[0]-.003)-R[:,2]*.004
    if anchor is not None and t>=105:grip,R=anchor
    if t<60:return grip+R[:,0]*.035,R,False,'三维对齐杯柄外侧'
    if t<105:return grip,R,False,'进入杯柄抓持位置'
    if t<125:return grip,R,True,'固定方式闭合夹爪'
    return grip+np.array([0,0,.065]),R,True,'保持抓持并抬升'
