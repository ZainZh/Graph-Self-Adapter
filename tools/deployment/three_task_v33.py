#!/usr/bin/env python3
"""V32 automatic execution plus independent continuous simulation RGB preview."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import three_task_v32 as base
import three_task_v26
three_task_v26.DEFAULT_REGISTRY=HERE.parents[1]/'evaluations/live_preview_v33/REGISTRY.json'
from task_stream_view_v33 import install
install()
from prove.xtrainer.agent import write,digest
import three_task_v26_operator_scope as scopes
original=scopes.new_operator_scope

def scope(*a,**kw):
    result=original(*a,**kw);p=Path(result['scope']);d=json.loads(p.read_text())
    d['display_preview']={'RGB_only':True,'target_fps':10,'private_state_snapshot_for_graphics_only':True,'writes_to_physics':False,'enters_controller_or_judge':False,'segmentation_and_depth_rendering':False}
    d['operator_source_bindings'] += [{'executed_path':str(f),'sha256':digest(f)} for f in (Path(__file__).resolve(),HERE/'sim_preview_v33.py',HERE/'task_stream_view_v33.py')]
    write(p,d);result['scope_sha256']=digest(p);return result
scopes.new_operator_scope=scope
if __name__=='__main__':base.main()
