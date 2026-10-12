#!/usr/bin/env python3
"""Additive bounded sensor repair runner; never modify initial source bytes."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from tools.task_graph import twin_sensor_rsi as initial
from prove.task_graph.twin_sensor_adapter_repair1 import repair_world_class
from prove.task_graph.twin_sensor_judge_repair1 import RepairConfig

REPAIR_SOURCES=('prove/task_graph/twin_sensor_judge_repair1.py','prove/task_graph/twin_sensor_adapter_repair1.py',
                'tools/task_graph/twin_sensor_rsi_repair1.py','tools/task_graph/twin_sensor_packet_replay_repair1.py',
                'tools/task_graph/twin_sensor_validate_repair1.py')


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if '--config' not in argv or '--contract' not in argv:
        raise ValueError('explicit reviewed repair config and contract paths required')
    initial.TASK_CANDIDATES={k:v for k,v in initial.TASK_CANDIDATES.items() if k!='sorting_cube_bimanual'}
    initial.JudgeConfig=RepairConfig
    initial.sensor_world_class=repair_world_class
    original_validate=initial.validate_execution
    def validate(scope,freeze_path):
        freeze=original_validate(scope,freeze_path)
        if freeze.get('execution_variant')!='sensor_only_repair1':
            raise ValueError('repair execution variant not frozen')
        if not all((REPO/path).resolve() in freeze['_validated_bound_paths'] for path in REPAIR_SOURCES):
            raise ValueError('missing actual repair source binding')
        return freeze
    initial.validate_execution=validate
    original_install=initial.install_sensor_proxy
    def install(*args,**kwargs):
        provenance=dict(kwargs.get('provenance') or {})
        provenance.update({str(REPO/path):initial.digest(REPO/path) for path in REPAIR_SOURCES})
        provenance['execution_variant']='sensor_only_repair1'
        kwargs['provenance']=provenance
        worlds=original_install(*args,**kwargs)
        runner=args[0] if args else kwargs['runner']
        dump=runner.dump_json
        def write(path,report):
            if report.get('schema')=='graph-self-adapter.twin-run.v1':
                report['execution_variant']='sensor_only_repair1'
                report['sensor_repair']='provisional obstruction/vicinity; aperture compliance and centroid motion diagnostics, no physical contact certification'
            return dump(path,report)
        runner.dump_json=write
        return worlds
    initial.install_sensor_proxy=install
    return initial.main(argv)


if __name__=='__main__':main()
