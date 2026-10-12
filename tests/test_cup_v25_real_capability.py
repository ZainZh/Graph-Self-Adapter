from pathlib import Path
import sys
import numpy as np
import pytest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from prove.xtrainer.io import RobotSample, SafetyError
from cup_v25_real_capability_sim_io import SimRobot, BaseSimRobot

def test_sim_aperture_not_available_like_real_decoder():
    sample=RobotSample(10,np.zeros(6),np.zeros(6),np.zeros(3),np.eye(3),.02,'sim_private_jaw')
    robot=SimRobot.__new__(SimRobot)
    with patch.object(BaseSimRobot,'samples',return_value=[sample,sample]):
        rows=robot.samples()
    assert all(s.aperture is None for s in rows)
    assert rows[0].q is sample.q and rows[0].tcp is sample.tcp

def test_sim_rejects_same_bad_public_commands_as_native_rpc():
    robot=SimRobot.__new__(SimRobot)
    with pytest.raises(SafetyError):robot.command(np.zeros((2,6)),[1,2],[True,True])
    with pytest.raises(SafetyError):robot.command(np.zeros((2,6)),[1,1],[False,False])
