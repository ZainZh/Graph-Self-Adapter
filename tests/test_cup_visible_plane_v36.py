import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_visible_plane_v36 import VisibleTableVision
from prove.xtrainer.io import SafetyError

def setup(tmp_path):
    v=VisibleTableVision.__new__(VisibleTableVision);v.evidence_root=tmp_path;v.stage_context=lambda:{'stage_id':'white.approach'}
    K=np.array([[120.,0,80],[0,120.,60],[0,0,1]]);depth=np.ones((120,160));T=np.eye(4);box=[30,40,80,90]
    v._table_plane(depth,K,T,box);return v,depth,K,T,box

def test_occluded_old_roi_reselects_measured_current_plane(tmp_path):
    v,d,K,T,box=setup(tmp_path);d[40:90,30:80]=.7
    cloud,mask=v._table_plane(d,K,T,box)
    assert np.allclose(cloud[:,2],1.);assert not mask[40:90,30:80].any()
    assert len(cloud)>=12
    assert v.table_relocations[-1]['fresh_current_depth_refit']
    assert (tmp_path/'TABLE_ROI_RELOCATIONS.json').exists()

def test_no_current_table_cannot_return_cached_cloud(tmp_path):
    v,d,K,T,box=setup(tmp_path);d[:]=.7
    with pytest.raises(SafetyError,match='no fresh visible tabletop'):
        v._table_plane(d,K,T,box)

def test_invalid_initial_roi_cannot_create_prior(tmp_path):
    v=VisibleTableVision.__new__(VisibleTableVision);d=np.zeros((30,30))
    with pytest.raises(SafetyError):v._table_plane(d,np.eye(3),np.eye(4),[0,0,20,20])
    assert not hasattr(v,'validated_table_prior')
