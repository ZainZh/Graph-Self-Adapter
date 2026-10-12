import json
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
import trajectory_white_runtime_v18 as m
from stage_trajectory import Limits


def bare():
    x=m.TrajectoryWhiteExecutor.__new__(m.TrajectoryWhiteExecutor);x.lower_contact_tolerance=.003;x.grasp_tolerance=.002
    x.stage={'cycle_id':'white_direct','phase':'Lower'}
    x.trajectory_limits=Limits([-1]*6,[1]*6,[-1]*3,[1]*3,position_tolerance=.001)
    return x


def test_only_white_lower_gets_contact_pose_tolerance():
    x=bare();goal=(np.zeros(3),np.eye(3));pose=(np.array([.00294,0,0]),np.eye(3))
    assert x._reached(goal,pose)
    for phase in ('Approach','Lift','Raise','Move','Retreat','Grasp'):
        x.stage['phase']=phase;assert not x._reached(goal,pose)
    x.stage['phase']='Grasp';assert x._reached(goal,(np.array([.0018,0,0]),np.eye(3)))


def test_stream_limits_lower_only_and_restored_on_failure(monkeypatch):
    x=bare();original=x.trajectory_limits;observed=[]
    def command(self,stage,*a,**k):observed.append(self.trajectory_limits.position_tolerance);raise m.SafetyError('stub dispatch failure')
    monkeypatch.setattr(m.V17Executor,'command',command)
    for phase in ('Lower','Grasp','Move'):
        x.stage['phase']=phase
        with pytest.raises(m.SafetyError,match='stub'):x.command(x.stage)
        assert x.trajectory_limits is original
    assert observed==[.003,.001,.001]


def test_owner_config_mapping_without_devices(monkeypatch):
    cfg=json.loads((ROOT/'evaluations/stage_trajectory_v18/CONFIG_PROPOSED.json').read_text())['runtime']
    monkeypatch.setattr(m.V17Executor,'__init__',lambda self,*a,**k:setattr(self,'cfg',cfg))
    x=m.TrajectoryWhiteExecutor();assert x.lower_contact_tolerance==.003
