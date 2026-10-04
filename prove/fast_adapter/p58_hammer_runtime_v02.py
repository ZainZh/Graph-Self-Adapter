"""Frozen P58 held-hammer geometry runtime and shared position controller."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import numpy as np
import torch

from prove.fast_adapter.p58_geometry_model import GeometryAdapterConfig,LowRankRayAdapter
from prove.fast_adapter.p58_hammer_readout_v02 import (
    load_package, predict_measurement, ray_system,
)

ARMS=("fixed","online","direct","mean_geometry","train_mean")


def _sha(path):return hashlib.sha256(open(path,"rb").read()).hexdigest()


class HammerGeometryRuntime:
    def __init__(self,model,readout,representation,arm="online",device="cpu",training_mean=None):
        if representation not in ("absolute_positions","role_relative") or arm not in ARMS:
            raise ValueError("Unknown P58 hammer runtime")
        self.model=model.to(device).eval().requires_grad_(False);self.readout=readout
        arm="mean_geometry" if arm=="train_mean" else arm
        self.representation=representation;self.arm=arm;self.device=torch.device(device)
        self.training_mean=None if training_mean is None else np.asarray(training_mean,np.float32).reshape(2,3)
        if arm=="mean_geometry" and self.training_mean is None:raise ValueError("Mean arm requires train mean")
        self.reset()

    def reset(self):
        self.z=self.model.z0.detach().to(self.device)[None].clone();self.history=[];self.t=0
        self.pushing=False;self.frozen_target=None;self.records=[]

    @staticmethod
    def _canonical(estimate,frame,representation):
        value=np.asarray(estimate,np.float32).reshape(2,3).copy()
        if representation=="absolute_positions":
            eef=np.asarray(frame["eef_transform"],np.float32)
            value[0]=eef[:3,:3].T@(value[0]-eef[:3,3])
        return value

    @staticmethod
    def _world(canonical,eef):
        value=np.asarray(canonical,np.float32).reshape(2,3).copy();T=np.asarray(eef,np.float32)
        value[0]=T[:3,:3]@value[0]+T[:3,3];return value

    def _support(self):
        rows=self.history[-8:]
        # ray_system is [role,view,...]; the shared core expects
        # [batch,support,point,...].  Each time/view pair is one support row.
        A=np.stack([r[0] for r in rows]).transpose(0,2,1,3,4).reshape(-1,2,3,3)
        C=np.stack([r[1] for r in rows]).transpose(0,2,1,3).reshape(-1,2,3)
        valid=np.stack([r[2] for r in rows]).transpose(0,2,1).reshape(-1,2)
        return (torch.as_tensor(A[None],dtype=torch.float32,device=self.device),
                torch.as_tensor(C[None],dtype=torch.float32,device=self.device),
                torch.as_tensor(valid[None],dtype=torch.bool,device=self.device))

    def observe(self,frame):
        uv,estimate=predict_measurement(self.readout,frame)
        g=self._canonical(estimate,frame,self.representation)
        A,C,valid=ray_system(uv,frame,"role_relative")
        if self.pushing:valid[1]=False
        self.history.append((A,C,valid));self.history=self.history[-8:]
        gt=torch.as_tensor(g[None],dtype=torch.float32,device=self.device)
        before=self.z.clone();updates=0
        if self.arm=="online" and self.t>=2:
            with torch.enable_grad():self.z,losses=self.model.adapt(gt,self.z,*self._support(),create_graph=False)
            self.z=self.z.detach();updates=len(losses[0])
        elif self.arm=="direct" and self.t>=2:
            with torch.no_grad():self.z=self.model.direct(gt,self.model.z0[None].to(self.device),*self._support())
            updates=1
        with torch.no_grad():decoded=self.model.decode(gt,self.z)[0].cpu().numpy()
        if self.arm=="mean_geometry":decoded=self.training_mean.copy()
        world=self._world(decoded,frame["eef_transform"])
        delta=world[1]-world[0];pre_y=world[1,1]-.10
        if not self.pushing and np.linalg.norm(delta[[0,2]])<.013 and abs(pre_y-world[0,1])<.025:
            self.pushing=True;self.frozen_target=world[1].copy()
        if self.pushing:world[1]=self.frozen_target
        action=hammer_position_controller(world,self.pushing)
        record=dict(t=self.t,arm=self.arm,representation=self.representation,pushing=self.pushing,
            g=g.tolist(),decoded_canonical=decoded.tolist(),estimated_world=world.tolist(),
            z=self.z.cpu().tolist(),z_delta=float((self.z-before).norm()),updates=updates,
            valid_rays=int(valid.sum()),action=action.tolist())
        self.records.append(record);self.t+=1
        return action,record

    step=observe


def hammer_position_controller(world_positions,pushing):
    """Shared frozen relation controller using estimated centers only."""
    head,target=np.asarray(world_positions,np.float32).reshape(2,3)
    goal=target.copy()
    if pushing:goal[1]+=.055
    else:goal[1]-=.10
    delta=goal-head
    if not pushing and abs(delta[2])>.008:
        delta=np.array([0.,0.,delta[2]],np.float32)
    return np.r_[np.clip(delta*15.,-1.,1.),.9].astype(np.float32)


def make_runtime(checkpoint,readout_path,representation=None,mode="online",device="cpu"):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if payload.get("schema")!="p58.hammer-geometry-model.v2" or payload.get("readout_sha256")!=_sha(readout_path):
        raise ValueError("P58 hammer checkpoint/readout binding mismatch")
    if representation is not None and representation!=payload.get("representation"):
        raise ValueError("P58 hammer requested representation/checkpoint mismatch")
    allowed={field.name for field in fields(GeometryAdapterConfig)}
    config=GeometryAdapterConfig(**{k:v for k,v in payload["config"].items() if k in allowed})
    model=LowRankRayAdapter(payload["state_dict"]["B"],config);model.load_state_dict(payload["state_dict"])
    readout=load_package(readout_path,payload["readout_sha256"])
    mean=payload["training_mean_canonical"]
    if torch.is_tensor(mean):mean=mean.numpy()
    return HammerGeometryRuntime(model,readout,payload["representation"],mode,device,mean)
