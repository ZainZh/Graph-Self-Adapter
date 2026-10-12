import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_v4r2',ROOT/'tools/deployment/handle_local_driver_v4r2.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_relative_lift_does_not_treat_visible_low_surface_as_actual_bottom():
    cloud=np.c_[np.zeros(40),np.zeros(40),np.linspace(.01,.09,40)];table=cloud.copy();table[:,2]=0.
    vision=SimpleNamespace(cloud=lambda role:cloud if role=='white_mug' else table,latest_observation_stamp=10.,raw_camera=SimpleNamespace(last_physical_time=5.))
    prior=driver.lock_observed_lift_prior(vision)
    assert np.isclose(prior['requested_rise_m'],.35*prior['observed_height_m'])
    assert prior['requested_rise_m']>.35*prior['observed_height_m']-prior['observed_low_z_m']
    vision.cloud=lambda role: (_ for _ in ()).throw(RuntimeError('currently occluded'))
    actual=(np.array([.2,.3,.1]),np.eye(3));goal,record=driver.lift_from_locked_prior(actual,prior)
    assert np.isclose(goal[0][2]-.1,.35*prior['observed_height_m']) and record['remeasured_after_grasp'] is False
