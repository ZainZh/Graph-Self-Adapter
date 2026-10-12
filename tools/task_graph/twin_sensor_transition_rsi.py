#!/usr/bin/env python3
"""Sensor Transition v04, fixed frozen A2 knife/cup controllers only."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from tools.task_graph import twin_sensor_rsi as initial
from prove.task_graph.twin_sensor_judge_repair1 import RepairConfig
from tools.task_graph.twin_sensor_rsi_repair1 import REPAIR_SOURCES
from prove.task_graph.twin_sensor_transition_adapter import transition_world_class,routed_holding_contract
SOURCES=('prove/task_graph/twin_sensor_transition.py','prove/task_graph/twin_sensor_transition_adapter.py',
         'tools/task_graph/twin_sensor_transition_rsi.py','tools/task_graph/twin_sensor_transition_validate.py')


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if '--config' not in argv or '--contract' not in argv:raise ValueError('explicit v04 config/contract required')
    initial.TASK_CANDIDATES={k:v for k,v in initial.TASK_CANDIDATES.items() if k!='sorting_cube_bimanual'}
    initial.JudgeConfig=RepairConfig;initial.sensor_world_class=transition_world_class
    validate=initial.validate_execution
    def guard(scope,path):
        freeze=validate(scope,path)
        if freeze.get('execution_variant')!='sensor_transition_v04':raise ValueError('wrong execution variant')
        if not all((REPO/p).resolve() in freeze['_validated_bound_paths'] for p in (*SOURCES,*REPAIR_SOURCES)):
            raise ValueError('missing actual v04 source bindings')
        return freeze
    initial.validate_execution=guard
    install=initial.install_sensor_proxy
    def extend(*args,**kwargs):
        provenance=dict(kwargs.get('provenance') or {})
        provenance.update({str(REPO/p):initial.digest(REPO/p) for p in (*SOURCES,*REPAIR_SOURCES)});provenance['execution_variant']='sensor_transition_v04'
        kwargs['provenance']=provenance;worlds=install(*args,**kwargs)
        runner=args[0] if args else kwargs['runner']
        runner.holding_fact=lambda contract,raw,**kw:routed_holding_contract(
            {**(worlds[-1].sensor_stage or {}),'holding_contract':contract,
             'target':kw.get('target_role'),'goal_relation':kw.get('goal_relation')},raw)
        dump=runner.dump_json
        def write(path,report):
            if report.get('schema')=='graph-self-adapter.twin-run.v1':
                report.update(execution_variant='sensor_transition_v04',
                              transition_permission_is_strict_placement=False)
            return dump(path,report)
        runner.dump_json=write
        return worlds
    initial.install_sensor_proxy=extend
    return initial.main(argv)


if __name__=='__main__':main()
