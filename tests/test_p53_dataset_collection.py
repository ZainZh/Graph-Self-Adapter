import copy
import pytest
from tools.task_graph.collect_libero_p53_dataset import validate_conditions


def conditions():
    entries=[]
    for i in range(24):
        g,q=divmod(i,4);split='validation' if q==g%4 else 'train';job=f'job_{i:02d}'
        entries.append(dict(index=i,job_id=job,split=split,source_condition=dict(job_id=job,split=split,grasp_index=g,quadrant_index=q)))
    return entries


def test_preserves_original_validation_denominator_and_identity():
    entries=conditions();validate_conditions(entries)
    assert sum(e['split']=='validation' for e in entries)==6
    with pytest.raises(ValueError):validate_conditions(entries[:-1])
    wrong=copy.deepcopy(entries);wrong[0]['split']='train'
    with pytest.raises(ValueError):validate_conditions(wrong)
    replaced=copy.deepcopy(entries);replaced[2]=copy.deepcopy(replaced[1])
    with pytest.raises(ValueError):validate_conditions(replaced)
