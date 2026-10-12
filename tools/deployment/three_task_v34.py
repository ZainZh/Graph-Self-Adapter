#!/usr/bin/env python3
"""Continuous preview with ordinary task background ROI separated from entity tracking."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import three_task_v33 as base
import three_task_v26
three_task_v26.DEFAULT_REGISTRY=HERE.parents[1]/'evaluations/live_preview_v34/REGISTRY.json'
import three_task_v26_operator_scope as scopes
from prove.xtrainer.agent import write,digest
original=scopes.new_operator_scope

def scope(*a,**kw):
    result=original(*a,**kw);p=Path(result['scope']);s=json.loads(p.read_text())
    s['background_ROI_contract']='Ordinary-task table_patch retained as initial RGB ROI only, never SAM tracked entity; object/target freshness and original safety gates unchanged'
    s['operator_source_bindings'] += [{'executed_path':str(f),'sha256':digest(f)} for f in (Path(__file__).resolve(),HERE/'api_grounding_v34.py',HERE/'api_v34_apple_entry.py',HERE/'preview_v34_collect_apple_entry.py')]
    write(p,s);result['scope_sha256']=digest(p);return result
scopes.new_operator_scope=scope
if __name__=='__main__':base.base.main()
