"""Additive v04 facade; strict attachment and stage-goal judgment remain unchanged."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from prove.task_graph import twin_sensor_adapter as frozen_adapter
from prove.task_graph.twin_sensor_adapter_repair1 import repair_world_class
from prove.task_graph.twin_sensor_transition import support_transition_candidate, transition_holding_contract
STRICT_CONTRACT=frozen_adapter.sensor_holding_contract


def routed_holding_contract(stage,facts):
    return transition_holding_contract(stage,facts,STRICT_CONTRACT)


def transition_world_class(physics, **kwargs):
    references,seal_path=kwargs['references'],Path(kwargs['seal_path'])
    class EvaluatorBoundary(physics.TwinWorld):
        def finalize_evaluator(self,outcome):
            seal=json.loads(seal_path.read_text())
            seal.update(execution_variant='sensor_transition_v04',
                        transition_permissions=self.transition_records,
                        transition_permission_is_strict_placement=False)
            seal_path.write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n')
            self.sensor_prediction_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest()
            return super().finalize_evaluator(outcome)
    facade=SimpleNamespace(**{k:v for k,v in vars(physics).items() if not k.startswith('__')})
    facade.TwinWorld=EvaluatorBoundary
    Base=repair_world_class(facade,**kwargs)
    class TransitionWorld(Base):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw);self.transition_records=[]
        def _sensor_facts(self,role,arm,support=None,relation=None):
            facts=super()._sensor_facts(role,arm,support,relation)
            stage=self.sensor_stage or {}
            reference=references.get('stage_relations',{}).get(stage.get('stage_id'),{})
            anchor=references.get('handover_anchors',{}).get(stage.get('cycle_id'),{}).get('centroid_world_m',reference.get('target_end'))
            value,diagnostic=support_transition_candidate(self.sensor_judge,stage,facts,reference,
                observed_role=role,observed_target=support,handover_anchor=anchor)
            facts['support_transition_candidate']=value
            self.transition_records.append({'timestamp_sec':float(self.data.time),'stage_id':stage.get('stage_id'),
                                            'value':value,'diagnostic':diagnostic})
            return facts
    # In-memory router shared by frozen preflight method; no on-disk mutation.
    frozen_adapter.sensor_holding_contract=routed_holding_contract
    return TransitionWorld
