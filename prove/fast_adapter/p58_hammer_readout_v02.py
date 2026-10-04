"""P58 hammer V2 readout with public predicted-UV image-boundary gating."""
from __future__ import annotations
import numpy as np
from prove.fast_adapter.p58_hammer_readout import *
from prove.fast_adapter.p58_hammer_readout import ray_system as _ray_system_v1
from prove.fast_adapter.p58_hammer_observation import RESOLUTION


def ray_system(uv, frame, representation):
    """Build rays, excluding only publicly predicted out-of-image points."""
    A,C,valid=_ray_system_v1(uv,frame,representation)
    value=np.asarray(uv,dtype=np.float64)
    if value.shape!=(2,2,2):raise ValueError("Expected [view,role,uv]")
    inside=((value[...,0]>=0.)&(value[...,0]<RESOLUTION)&
            (value[...,1]>=0.)&(value[...,1]<RESOLUTION))
    return A,C,valid&inside.T
