#!/usr/bin/env python3
"""Isolated V16 bindings and loader over unchanged V15 thin simulator entry."""
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (HERE,REPO):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import direct_white_entry as base
from prove.xtrainer.agent import digest
from prove.xtrainer.io import SafetyError

TASK=base.TASK
VARIANT='white_stage_trajectory_v16'
_original_bindings=base.bindings
_original_hash=base.runtime_hash
_original_load=base.load
SOURCES=(HERE/'stage_trajectory.py',HERE/'trajectory_white_runtime.py',HERE/'trajectory_entry.py',REPO/'tests/test_stage_trajectory.py',REPO/'tests/test_trajectory_white_runtime.py')

def runtime_hash():return hashlib.sha256((_original_hash()+''.join(digest(path) for path in SOURCES)).encode()).hexdigest()

def bindings(config_path,cfg,mode):
    config=cfg['runtime']['trajectory']
    for key in ('rate_hz','velocity_rad_sec','acceleration_rad_sec2','maximum_lateness_sec','settle_seconds','grasp_position_tolerance_m'):
        if not isinstance(config[key],(int,float)) or not 0<float(config[key])<float('inf'):raise SafetyError('positive finite trajectory setting required: '+key)
    if config['grasp_position_tolerance_m']>.002:raise SafetyError('white Grasp tolerance cannot exceed2mm')
    report,effective,bundle,bound=_original_bindings(config_path,cfg,mode)
    bound.update({str(path.resolve()):digest(path) for path in SOURCES})
    return report,effective,bundle,bound

def load(path,name):
    selected=HERE/'trajectory_white_runtime.py' if Path(path).name=='direct_white_runtime.py' else path
    return _original_load(selected,name)

def install():
    base.bindings=bindings;base.runtime_hash=runtime_hash;base.load=load;base.VARIANT=VARIANT

def guard(*args,**kwargs):return base.guard(*args,**kwargs)

def main(argv=None):
    install();return base.main(argv)

if __name__=='__main__':main()
