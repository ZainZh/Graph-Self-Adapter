"""Repair facade preserving initially frozen servo and evaluator isolation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from prove.task_graph.twin_sensor_adapter import sensor_world_class
from prove.task_graph.twin_sensor_judge_repair1 import RepairJudge


def repair_world_class(physics, **kwargs):
    if kwargs.get('judge_mode') != 'sensor_only':
        raise ValueError('repair is sensor-only; initial GT control is not rerun')
    config, seal_path = kwargs['config'], Path(kwargs['seal_path'])
    class EvaluatorBoundary(physics.TwinWorld):
        def finalize_evaluator(self, outcome):
            # Parent proxy has sealed and enabled its evaluator-only context;
            # tag actual additive execution before original GT evaluator runs.
            seal = json.loads(seal_path.read_text())
            seal['execution_variant'] = 'sensor_only_repair1'
            seal['repair_claim'] = 'provisional visible obstruction proxy, not contact or rigid attachment'
            seal_path.write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n')
            self.sensor_prediction_sha256 = hashlib.sha256(seal_path.read_bytes()).hexdigest()
            return super().finalize_evaluator(outcome)
    facade = SimpleNamespace(**{name:value for name,value in vars(physics).items() if not name.startswith('__')})
    facade.TwinWorld = EvaluatorBoundary
    Base = sensor_world_class(facade, **kwargs)
    class RepairWorld(Base):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.sensor_judge = RepairJudge(config)
    return RepairWorld
