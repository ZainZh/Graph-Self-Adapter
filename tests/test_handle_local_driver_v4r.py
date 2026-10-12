import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from prove.xtrainer.io import SafetyError
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_v4r',ROOT/'tools/deployment/handle_local_driver_v4r.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_lift_uses_locked_pregrasp_size_when_table_mask_becomes_occluded():
    cup=np.c_[np.zeros(40),np.zeros(40),np.linspace(0,.08,40)];table=cup.copy();table[:,2]=0.
    vision=SimpleNamespace(cloud=lambda role:cup if role=='white_mug' else table,clock=lambda:20.,latest_observation_stamp=10.,raw_camera=SimpleNamespace(last_physical_time=5.))
    prior=driver.lock_observed_lift_prior(vision)
    def missing(role):raise SafetyError('observed table completely occluded')
    vision.cloud=missing
    actual=(np.array([.2,.3,.1]),np.eye(3));goal,record=driver.lift_from_locked_prior(actual,prior)
    assert np.isclose(goal[0][2]-actual[0][2],prior['requested_rise_m']) and np.array_equal(goal[1],actual[1])
    assert record['static_size_table_prior'] and record['reused_observation_prior_not_fresh_measurement']
    assert record['observation_stamp']==10. and record['physical_time_sec']==5. and record['remeasured_after_grasp'] is False


def test_missing_initial_cup_size_is_not_fabricated():
    vision=SimpleNamespace(cloud=lambda role:np.zeros((0,3)))
    with pytest.raises(SafetyError,match='unknown'):driver.lock_observed_lift_prior(vision)
