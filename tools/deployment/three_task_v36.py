#!/usr/bin/env python3
"""Cup current-visible table plane repair plus same MJPEG and automatic API pipeline."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import three_task_v35 as base
import three_task_v26
three_task_v26.DEFAULT_REGISTRY=HERE.parents[1]/'evaluations/cup_plane_v36/REGISTRY.json'
import three_task_v26_operator_scope as scopes
from prove.xtrainer.agent import write,digest
original=scopes.new_operator_scope

def scope(*a,**kw):
    result=original(*a,**kw);p=Path(result['scope']);s=json.loads(p.read_text())
    s['table_observer_contract']='Initial validated visible plane selects fresh alternate RGBD patches; refit preserves original 3mm/80percent/horizontal/extent gates. No prior cloud returned as current observation; background centroid excluded from entity drift guard.'
    s['operator_source_bindings'] += [{'executed_path':str(f),'sha256':digest(f)} for f in (Path(__file__).resolve(),HERE/'api_v36_cup_entry.py',HERE/'cup_visible_plane_v36.py')]
    write(p,s);result['scope_sha256']=digest(p);return result
scopes.new_operator_scope=scope
if __name__=='__main__':base.base.base.base.main()
