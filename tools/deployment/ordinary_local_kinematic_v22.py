#!/usr/bin/env python3
"""Local ordinary derivative: exact kinematic FK and optional unused table role."""
import inspect,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for p in (HERE,REPO):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
import multitask_ordinary_entry_v21 as base
import multitask_one_command_v21 as launcher
import multitask_ordinary_runtime as runtime
from prove.xtrainer.io import SafetyError
from multitask_ordinary_route import role_contract as original_roles
_installed=False

def optional_roles(graph,refs):
 result=original_roles(graph,refs)
 if graph['task']!='sort_two_spoon':raise SafetyError('local candidate is spoon-specific')
 actual={s['object'] for s in graph['stages']}|{s['target'] for s in graph['stages']}
 if 'table_patch' not in actual:
  result['SAM_roles']=[r for r in result['SAM_roles'] if r!='table_patch'];result['optional_SAM_roles']=['table_patch']
 return result

def backend_contract():
 import kinematic_sim_robot_io as io
 for name in ('SimRobot','SimRGBDCamera','SimStop'):
  if not callable(getattr(io,name,None)):raise SafetyError('missing simulator backend '+name)
 inspect.signature(io.SimRobot).bind(scene_run=None,physics_source=None,bundle={},T_left_from_right=None,initial_q=None)
 inspect.signature(io.SimRGBDCamera).bind(None,width=640,height=360)
 inspect.signature(io.SimStop).bind(None,"left")
 return io

def install():
 global _installed
 if _installed:return
 io=backend_contract();original=base.backend.load
 def load(path,name):return io if Path(path).name=='sim_robot_io.py' else original(path,name)
 base.backend.load=load;base.role_contract=optional_roles;runtime.role_contract=optional_roles
 closure=list((Path('/home/zheng/github/SAM2_streaming')/'sam2').rglob('*.py'))+list((Path('/home/zheng/github/SAM2_streaming')/'sam2').glob('_C*.so'))
 if not any(p.suffix=='.so' for p in closure):raise SafetyError('native SAM extension absent')
 base.SOURCES=base.SOURCES+(Path(__file__),HERE/'fair_sim_robot_io.py',HERE/'kinematic_sim_robot_io.py',REPO/'tests/test_ordinary_local_kinematic_v22.py')+tuple(sorted(closure))
 _installed=True

def main(argv=None):
 install();return launcher.main(argv)
if __name__=='__main__':main()
