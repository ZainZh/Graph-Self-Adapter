import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'tools/deployment/readonly_entry_check.py'
spec=importlib.util.spec_from_file_location('_readonly_entry_parameters',p);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def boxes():return {'white_mug':[0,0,2,2],'red_mug':[2,0,4,2],'box':[0,2,4,4]}


def test_explicit_boxes_required_and_invalid_values_rejected():
    assert entry.validate_boxes(json.dumps(boxes()))==boxes()
    for invalid in ({'red_mug':[0,0,2,2]},{**boxes(),'box':[0,0,0,2]},{**boxes(),'box':[0,0,float('nan'),2]}):
        with pytest.raises(ValueError):entry.validate_boxes(json.dumps(invalid))


def test_cached_timestamp_is_preserved_and_never_promoted_to_live(tmp_path):
    p=tmp_path/'cached.npz';np.savez(p,rgb=np.zeros((4,4,3),np.uint8),depth_m=np.ones((4,4)),
                                  camera_matrix=np.eye(3),timestamp_monotonic_sec=12.25,serial=38968158)
    _,_,_,meta=entry.read_cached_rgbd(p,boxes())
    assert meta['cached_timestamps']['timestamp_monotonic_sec']==12.25
    assert meta['observation_cached'] and not meta['live_admission'] and not meta['timestamps_rebased']
    wrong={**boxes(),'box':[0,2,5,4]}
    with pytest.raises(ValueError,match='outside'):entry.read_cached_rgbd(p,wrong)
