#!/usr/bin/env python3
"""Single cube ordinary interface and automatic sparse-witness API guard."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for p in (HERE,REPO):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--task',choices=('collect_single_cube',),default='collect_single_cube');p.add_argument('--backend',choices=('sim','real'),default='sim');p.add_argument('--check',action='store_true');a=p.parse_args(argv)
    root=REPO/'evaluations/task_expansion_v45/cube_variant';owner=root/'collect_cubes/state_pose_separation_v01'
    if a.backend=='real':
        if not a.check:p.error('offline checks only; no real devices')
        from three_task_v26_public_io import offline_contract_check,real_constructor_check
        r=offline_contract_check('collect_cubes');r['real_constructor']=real_constructor_check(json.loads((owner/'CONFIG_PROPOSED.json').read_text())['runtime']);print(json.dumps(r,indent=2));return r
    import cube_single_v45_entry as entry
    args=['--task','collect_cubes','--owner-root',str(root),'check' if a.check else 'sim','--config',str(owner/'CONFIG_PROPOSED.json'),'--scope',str(owner/'TRIAL_SCOPE.json')]
    if a.check:return entry.main(args)
    from task_stream_view_v33 import install
    install()
    from task_stream_view_v35 import install as labels
    labels()
    import task_live_view_v27
    view=task_live_view_v27.TaskLiveView(a.task,18741);view.set_owner(owner);print(json.dumps({'task':a.task,'live_view_url':view.start()}),flush=True)
    try:
        r=entry.main(args)
        if isinstance(r,dict) and 'post_seal_private_evaluation' in r:
            run=max((owner/'runs').iterdir(),key=lambda p:p.stat().st_mtime);e=r['post_seal_private_evaluation'];c=e['task_outcome']['checks'];passed=bool(r['graph_completed'] and c.get('yellow_cube_grasped') and c.get('yellow_cube_inside_bowl') and e['numerical_safety']['passed'])
            report={'logical_task':'collect_single_cube','scene_task':'collect_cubes','graph_completed':r['graph_completed'],'failure':r['failure'],'completed_nodes':len(r['stage_trace']),'task_goal_passed':passed,'five_cube_pass_claim':False,'numerical_safety_passed':e['numerical_safety']['passed'],'private_evaluation_after_seal':True};(run/'TASK_VARIANT_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
        return r
    finally:view.finished=True;view.refresh();view.close()
if __name__=='__main__':main()
