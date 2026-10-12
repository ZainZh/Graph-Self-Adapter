import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from prove.xtrainer.io import SafetyError
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_v4r3',ROOT/'tools/deployment/handle_local_driver_v4r3.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_goal_represents_observed_envelope_clearance_even_when_cup_rotates():
    rng=np.random.default_rng(3);cloud=rng.uniform([-.04,-.03,.035],[.04,.03,.115],(200,3));table=np.c_[np.zeros(30),np.zeros(30),np.full(30,.03)]
    vision=SimpleNamespace(cloud=lambda role:cloud if role=='white_mug' else table,latest_observation_stamp=10.,raw_camera=SimpleNamespace(last_physical_time=5.))
    prior=driver.lock_grasp_envelope(vision,np.array([-.06,0,.09]));prior['robot_pad_midpoint_TCP_local_m']=[0,0,-.032]
    actual=(np.array([-.06,0,.06]),np.diag([1.,-1.,-1.]));goal,record=driver.lift_from_locked_envelope(actual,prior)
    grasp_height=(goal[0]+goal[1]@np.array(prior['robot_pad_midpoint_TCP_local_m']))[2]
    assert np.isclose(grasp_height,prior['observed_table_z_m']+prior['visible_extent_radius_m']+.25*prior['observed_visible_height_m'])
    assert record['partial_view_estimate_not_full_object_bound'] and 0<record['requested_rise_m']<=.12
    vision.cloud=lambda role: (_ for _ in ()).throw(RuntimeError('currently occluded'))
    assert driver.lift_from_locked_envelope(actual,prior)[1]['remeasured_after_grasp'] is False


def test_envelope_target_over_original_lift_limit_fails_without_clipping():
    prior={'robot_pad_midpoint_TCP_local_m':[0,0,0],'goal_grasp_point_height_m':.3}
    with pytest.raises(SafetyError,match='.12m'):driver.lift_from_locked_envelope((np.zeros(3),np.eye(3)),prior)
