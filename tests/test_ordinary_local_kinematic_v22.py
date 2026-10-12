import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'));sys.path.insert(0,str(ROOT))
import ordinary_local_kinematic_v22 as entry

def test_real_backend_exports_and_constructor_contract():
 io=entry.backend_contract()
 assert io.SimRobot.__name__=='KinematicFairSimRobot'
 assert all(callable(getattr(io,n)) for n in ('SimRobot','SimRGBDCamera','SimStop'))

def test_actor_and_target_required_optional_table_only():
 from prove.xtrainer.multitask_bundle import load_bundle
 from multitask_ordinary_route import ordinary_route
 b=load_bundle('sort_two_spoon');roles=entry.optional_roles(ordinary_route(b['graph']),b['refs'])
 assert set(('black_spoon','orange_spoon','plate'))<=set(roles['SAM_roles'])
 assert 'table_patch' not in roles['SAM_roles']
