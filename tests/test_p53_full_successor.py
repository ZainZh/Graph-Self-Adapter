import json
import pytest
from tools.task_graph.p53_full_successor import closed_gate,sha


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def test_closed_gate_keeps_scientific_failures_and_rejects_engineering_abort(tmp_path):
    write(tmp_path/'COMPLETE.json',dict(expected_cells=36,closed_queue_attempts=36,sealed_scientific_results=36,engineering_aborts=0))
    attempts=[]
    for i in range(36):
        job=f'cell_{i:02d}';folder=tmp_path/'eval'/job/'episode';write(folder/'BINDING.json',{})
        write(folder/'SEAL.json',dict(binding_sha256=sha(folder/'BINDING.json'),files={}))
        write(folder/'RESULT.json',dict(cell=dict(index=i),system_success=False,seal_sha256=sha(folder/'SEAL.json')))
        attempts.append(dict(job=dict(id=job),returncode=0,execution_status='finished'))
    write(tmp_path/'eval/QUEUE_RESULT.json',dict(attempts=attempts))
    assert closed_gate(tmp_path)['cells']==36
    attempts[0]['returncode']=1;write(tmp_path/'eval/QUEUE_RESULT.json',dict(attempts=attempts))
    with pytest.raises(ValueError):closed_gate(tmp_path)
