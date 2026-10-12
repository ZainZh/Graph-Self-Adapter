#!/usr/bin/env python3
"""Parallel ten-task leaf dispatcher; no cup fallback or hidden device startup."""
import argparse,json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
TASKS=('collect_apple','collect_cubes','hanging_mug','open_drawer_and_place_object','put_knife_board_new','put_mug_on_box','sort_two_spoon','sorting_cube_bimanual','stack_cups','sweep_objects')
STATUS=REPO/'evaluations/multitask_parallel_v22/STATUS.json'


def listing():
    status=json.loads(STATUS.read_text());groups=status.get('groups',{});rows=[]
    for task in TASKS:
        group=next((group for group,tasks in groups.items() if task in tasks),'reference' if task=='put_mug_on_box' else 'unknown')
        candidates=[STATUS.parent/group/task/'STATUS.json',STATUS.parent/task/'STATUS.json']
        task_status=next((json.loads(p.read_text()) for p in candidates if p.is_file()),None)
        global_rows=status.get('tasks',{})
        if isinstance(global_rows,dict):task_status=global_rows.get(task,task_status)
        elif isinstance(global_rows,list):task_status=next((r for r in global_rows if r.get('task')==task),task_status)
        rows.append({'task':task,'group':group,'leaf':str(REPO/'task_packages/multitask_parallel_v22'/task/'run.sh'),'leaf_present':(REPO/'task_packages/multitask_parallel_v22'/task/'run.sh').is_file(),'qualification_from_parallel_STATUS':task_status if task_status is not None else {'state':'reference_V19_preserved' if task=='put_mug_on_box' and status.get('cup_V19_reference_preserved') else 'not_declared_in_STATUS','physical_hardware_qualification':False},'launch_file_presence_is_not_algorithm_success':True})
    return rows


def leaf_command(task,mode,extra=()):
    if task not in TASKS:raise ValueError('unknown/unqualified task; no default task fallback')
    if mode not in ('check','sim','shadow','real'):raise ValueError('explicit supported mode required')
    path=REPO/'task_packages/multitask_parallel_v22'/task/'run.sh'
    if not path.is_file():raise ValueError('specific task leaf missing; no fallback')
    return ['bash',str(path),mode,*extra]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('list','check','sim','shadow','real'),nargs='?',default='list');p.add_argument('--task');p.add_argument('--mode',choices=('check','sim','shadow','real'));a,extra=p.parse_known_args(argv)
    if a.action=='list' and not a.task:result=listing();print(json.dumps(result,indent=2));return result
    if not a.task:raise ValueError('explicit --task required')
    mode=a.mode or ('check' if a.action=='list' else a.action)
    return subprocess.run(leaf_command(a.task,mode,extra),check=True).returncode

if __name__=='__main__':main()
