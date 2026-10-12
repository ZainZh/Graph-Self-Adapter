#!/usr/bin/env python3
"""Cup current-visible table plane repair plus same MJPEG and automatic API pipeline."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import three_task_v35 as base
import three_task_v26
three_task_v26.DEFAULT_REGISTRY=HERE.parents[1]/'evaluations/cup_support_v43/REGISTRY.json'
import three_task_v26_operator_scope as scopes
from prove.xtrainer.agent import write,digest
original=scopes.new_operator_scope

def scope(*a,**kw):
    result=original(*a,**kw);p=Path(result['scope']);s=json.loads(p.read_text())
    s['support_goal_orientation_contract']='Yaw plus <=0.55rad roll/pitch norm; full visible convex boundary remains inside original box10mm margin and outside red, bottom20mm release clearance, original joint bounds'
    s['empty_release_nodes']=['c02.07b.clear_retreat','white_direct.08b.clear_retreat']
    s['empty_retreat_goal_solver']='Own bounded joints with free wrist orientation; endpoint pads clear observed cup top/projection, checked intermediate joint interpolation; original runtime path and safety checks retained'
    s['release_observability_contract']='After original c02 release/Wait, raise/retreat empty gripper using current mug height/extent and own CAD pad envelope before Complete. Real release remains API judged; no command implies success.'
    s['surface_visibility_contract']='Current SAM cup may contain 2-4 significant visible components. Each >=200 points, union>=70percent coverage, within initial visible cup bbox diagonal +10mm. Original final support/red exclusion and safety unchanged.'
    s['diagnostic_grasp_contract']='Grasp closure is not holding proof; only original Lift before API confirmation; no Move/Raise until positive fresh holding'
    s['table_observer_contract']='Initial validated visible plane selects fresh alternate RGBD patches; refit preserves original 3mm/80percent/horizontal/extent gates. No prior cloud returned as current observation; background centroid excluded from entity drift guard.'
    s['operator_source_bindings'] += [{'executed_path':str(f),'sha256':digest(f)} for f in (Path(__file__).resolve(),HERE/'api_v43_cup_entry.py',HERE/'cup_visible_plane_v36.py',HERE/'cup_probe_lift_v37_runtime.py',HERE/'cup_visible_surfaces_v39.py',HERE/'cup_clear_retreat_v40_runtime.py',HERE/'cup_clear_goalset_v41.py',HERE/'cup_both_clear_v42_runtime.py',HERE/'cup_support_goalset_v43.py')]
    write(p,s);result['scope_sha256']=digest(p);return result
scopes.new_operator_scope=scope
if __name__=='__main__':base.base.base.base.main()
