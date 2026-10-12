#!/usr/bin/env python3
"""Registry-selected task workflow; sim execution and real offline checks share public I/O contracts."""
import argparse
import importlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (REPO,HERE):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
DEFAULT_REGISTRY=REPO/'evaluations/three_task_migration_v25/REGISTRY_V26.json'
TASKS=('put_mug_on_box','collect_apple','collect_cubes')

def read_registry(path):
    registry=json.loads(path.read_text())
    if registry.get('schema')!='graph-self-adapter.task-execution-registry.v1' or set(registry.get('tasks',{}))!=set(TASKS):
        raise ValueError('exact three-task execution registry required')
    for row in registry['tasks'].values():
        if not re.fullmatch('[a-z][a-z0-9_]*',row['entry_module']):raise ValueError('invalid local entry module')
    return registry

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--backend',choices=('sim','real'),default='sim')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--experimental',action='store_true',help='explicitly execute a registered unqualified experiment when its default is disabled')
    parser.add_argument('--registry',type=Path,default=DEFAULT_REGISTRY)
    parser.add_argument('--real-config',type=Path,default=REPO/'evaluations/gemini_graph_v19/CONFIG_PROPOSED.json')
    args,remainder=parser.parse_known_args(argv)
    if args.list:
        registry=read_registry(args.registry)
        result={'registry':str(args.registry.resolve()),'tasks':registry['tasks'],'hardware_connections':0,'episodes_started':0}
        print(json.dumps(result,indent=2));return result
    if args.task is None:parser.error('--task required')
    if args.backend=='real':
        if not args.check:parser.error('real execution is not authorized in this work phase; --check opens no devices')
        if remainder:parser.error('real offline contract check has no execution arguments')
        from three_task_v26_public_io import offline_contract_check,real_constructor_check
        config=json.loads(args.real_config.read_text());config=config.get('runtime',config)
        result=offline_contract_check(args.task);result['real_constructor_check']=real_constructor_check(config)
        result['component_lifecycle']='robot and camera connect/close independently; BackendBundle connects each once'
        result['ready_flags']='disconnected/unready at constructor; camera ready after valid RGBD, robot ready only after fresh healthy public feedback'
        print(json.dumps(result,indent=2));return result
    row=read_registry(args.registry)['tasks'][args.task]
    if not args.check and row.get('default_execution_enabled') is False and not args.experimental:
        parser.error('this task has no qualified default; latest experiment is negative. Use --check, or explicitly --experimental for a new engineering validation')
    source=HERE/(row['entry_module']+'.py')
    if not source.is_file():parser.error('registered candidate entry is not ready: '+str(source))
    generated=None
    if not args.check and not any(value=='--scope' or value.startswith('--scope=') for value in remainder):
        from three_task_v26_operator_scope import new_operator_scope
        generated=new_operator_scope(args.task,row,args.registry)
        remainder+=['--scope',generated['scope']]
        if row['interface']=='ordinary_task_entry':
            remainder+=['--owner-root',generated['owner_root'],'--config',str(REPO/row['candidate_owner']/row.get('config_filename','CONFIG_PROPOSED.json'))]
    run=importlib.import_module(row['entry_module']).main
    if row['interface']=='cup_one_command':
        invocation=['sim']+(['--check'] if args.check else [])
    elif row['interface']=='ordinary_task_entry':
        invocation=['--task',args.task,'check' if args.check else 'sim']
    else:parser.error('unknown registry entry interface')
    return run(invocation+row.get("default_arguments",[])+remainder)

if __name__=='__main__':main()
