import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
import cup_visible_surfaces_v38 as surface
from prove.xtrainer.io import SafetyError

def cloud(gap):
    rng=np.random.default_rng(42)
    return np.r_[rng.normal([0,0,0],.001,(500,3)),rng.normal([0,0,gap],.001,(400,3))]

def test_two_coherent_observed_surfaces_are_retained(monkeypatch):
    monkeypatch.setattr(surface,'DIAMETER_BOUND',.09)
    data,report=surface.observed_surfaces(cloud(.035))
    assert len(data)==900;assert report['component_sizes']==[500,400]
    assert report['minimum_total_coverage']==.70

def test_unbounded_fragments_cannot_be_combined(monkeypatch):
    monkeypatch.setattr(surface,'DIAMETER_BOUND',.09)
    with pytest.raises(SafetyError,match='bounded instance'):
        surface.observed_surfaces(cloud(.2))

def test_failed_record_public_geometry_supports_hollow_surface_hypothesis(monkeypatch):
    root=Path(__file__).resolve().parents[1]/'evaluations/three_task_migration_v25/put_mug_on_box/current_entry_goalset_v01/operator_runs/20261008_165931_5fc57bfa/runs/20261008_165931_f0d544d9/SAM_MASK_EVIDENCE'
    if not root.exists():pytest.skip('local opened regression evidence')
    def points(index,role):
        mask=np.load(root/f'MASKS_{index:06d}.npz')[role];d=np.load(root/f'RGBD_{index:06d}.npz');y,x=np.nonzero(mask);z=d['depth_m'][y,x];K=d['K'];T=d['T'];return np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]@T[:3,:3].T+T[:3,3]
    bound=max(np.linalg.norm(np.ptp(points(0,r),axis=0)) for r in ('red_mug','white_mug'))+.01
    monkeypatch.setattr(surface,'DIAMETER_BOUND',bound)
    data,report=surface.observed_surfaces(points(170,'white_mug'))
    assert report['dominant_fraction']>=.70;assert len(data)>=1500
