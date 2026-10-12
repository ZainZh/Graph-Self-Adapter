#!/usr/bin/env python3
"""Scoped task expansion: common public I/O, automatic current RGB grounding and MJPEG."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for p in (HERE,REPO):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
TASKS={'put_knife_board_new':('put_knife_board_new',False,18742),'put_black_spoon_on_plate':('sort_two_spoon',True,18743)}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=tuple(TASKS),required=True);p.add_argument('--backend',choices=('sim','real'),default='sim');p.add_argument('--check',action='store_true');p.add_argument('--live-port',type=int)
    a=p.parse_args(argv);task,single,port=TASKS[a.task];owner=REPO/'evaluations/task_expansion_v45/knife_v46'/task
    if a.backend=='real':
        if not a.check:p.error('this phase supports only offline real interface checks; no devices opened')
        from three_task_v26_public_io import offline_contract_check,real_constructor_check
        cfg=json.loads((owner/'CONFIG_PROPOSED.json').read_text());report=offline_contract_check(task);report['real_constructor']=real_constructor_check(cfg['runtime']);print(json.dumps(report,indent=2));return report
    import generic_probe_v46_entry as entry
    entry.configure(task,single)
    args=['--task',task,'--owner-root',str(owner.parent),'check' if a.check else 'sim','--config',str(owner/'CONFIG_PROPOSED.json'),'--scope',str(owner/'TRIAL_SCOPE.json')]
    if a.check:return entry.main(args)
    from task_stream_view_v33 import install
    install()
    from task_stream_view_v35 import install as labels
    labels()
    import task_live_view_v27
    view=task_live_view_v27.TaskLiveView(a.task,a.live_port or port);view.set_owner(owner)
    print(json.dumps({'task':a.task,'live_view_url':view.start(),'hardware_connections':0}),flush=True)
    try:
        result=entry.main(args)
        if isinstance(result,dict) and 'post_seal_private_evaluation' in result:
            run=max((owner/'runs').iterdir(),key=lambda x:x.stat().st_mtime);evaluation=result['post_seal_private_evaluation'];checks=evaluation['task_outcome']['checks']
            # Exact simplified check mapping is frozen in the pre-run scope.
            scope=json.loads((owner/'TRIAL_SCOPE.json').read_text());names=scope.get('goal_checks')
            if not names:goal=evaluation['task_outcome']['passed']
            else:goal=all(checks.get(name) is True for name in names)
            report={'logical_task':a.task,'scene_task':task,'graph_completed':result['graph_completed'],'failure':result['failure'],'completed_nodes':len(result['stage_trace']),'goal_checks':checks if names is None else {n:checks.get(n) for n in names},'task_goal_passed':bool(result['graph_completed'] and goal and evaluation['numerical_safety']['passed']),'numerical_safety_passed':evaluation['numerical_safety']['passed'],'private_evaluation_after_seal':True,'hardware_motion_commands':result['hardware_motion_commands']}
            (run/'TASK_VARIANT_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
        return result
    finally:view.finished=True;view.refresh();view.close()
if __name__=='__main__':main()
