from types import SimpleNamespace
import torch
import pytest
from prove.fast_adapter.p68_book_runtime import BookReferenceState,ReferenceBookRuntime


def test_reference_locks_first_admitted_and_isolated_from_raw_drift():
    state=BookReferenceState('reference_fixed');raw=torch.arange(9.).float()
    assert state.select(raw,0,0,[])[0] is None
    base,event=state.select(raw,1,1,[SimpleNamespace(t=1)])
    assert event is None and state.reference_time==1
    raw.add_(100)
    later,event=state.select(raw,2,0,[])
    assert torch.equal(later,torch.arange(9.).float()) and event is None


def test_entry_and_continuous_share_first_original_eligible_event():
    once=BookReferenceState('entry_once');continuous=BookReferenceState('continuous')
    history=[SimpleNamespace(t=0)];raw=torch.zeros(9)
    for state in (once,continuous):
        assert state.select(raw,0,1,history)[1] is None
        assert state.select(raw,1,0,history)[1] is None
        assert state.select(raw+1,2,1,history+[SimpleNamespace(t=2)])[1]=='entry'
        assert state.select(raw+2,3,0,history)[1] is None
    assert once.select(raw,4,1,[SimpleNamespace(t=4)])[1] is None
    assert continuous.select(raw,4,1,[SimpleNamespace(t=4)])[1]=='continuous'


def test_frame_fixed_preserves_current_raw_and_reject_future_feedback():
    state=BookReferenceState('frame_fixed')
    for t in range(3):
        raw=torch.full((9,),float(t));base,event=state.select(raw,t,0,[])
        assert base is raw and event is None and state.reference is None
    with pytest.raises(ValueError,match='future'):state.select(raw,0,1,[SimpleNamespace(t=1)])


def test_nonfinite_reference_wait_and_lifecycle_exit_zero_archive():
    state=BookReferenceState('continuous')
    assert state.select(torch.full((9,),float('nan')),0,1,[SimpleNamespace(t=0)])[0] is None
    runtime=ReferenceBookRuntime.__new__(ReferenceBookRuntime)
    runtime.device='cpu';runtime.arm='continuous';runtime.epoch=-1;runtime.lifecycle=[]
    runtime.reset();runtime.z=torch.ones(9);runtime.exit()
    assert runtime.z.count_nonzero()==0 and runtime.anchor is None and not runtime.history
    assert runtime.reference.reference is None
    assert runtime.lifecycle[-1]['archived_z']==[1.]*9


def test_runner_keeps_original_physical_loop_and_success_evaluator():
    import ast
    from pathlib import Path
    def run(name):
        tree=ast.parse(Path(name).read_text())
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    old=run('tools/task_graph/p64_book_evaluate.py');new=run('tools/task_graph/p68_book_evaluate.py')
    # The entire action/physics/judge loop is identical: only runtime object construction differs outside it.
    old_loop=next(n for n in ast.walk(old) if isinstance(n,ast.While))
    new_loop=next(n for n in ast.walk(new) if isinstance(n,ast.While))
    assert ast.dump(old_loop,include_attributes=False)==ast.dump(new_loop,include_attributes=False)


def test_exit_has_no_phantom_enter_and_reset_starts_new_epoch():
    runtime=ReferenceBookRuntime.__new__(ReferenceBookRuntime)
    runtime.device='cpu';runtime.arm='entry_once';runtime.epoch=-1;runtime.lifecycle=[]
    runtime.reset();runtime.exit()
    assert [r['event'] for r in runtime.lifecycle]==['enter','exit']
    assert runtime.epoch==0 and not runtime.active
    with pytest.raises(ValueError,match='Explicit reset'):runtime.step({})
    runtime.reset()
    assert runtime.epoch==1 and runtime.active


def test_assignment_only_accepts_two_exact_complete_shards(tmp_path):
    import json
    from tools.task_graph.p68_book_evaluate import validate_assignments,ref
    files=[]
    for gpu in range(2):
        path=tmp_path/f'shard{gpu}.json'
        path.write_text(json.dumps(dict(shard=f'shard{gpu}',gpu=gpu,indices=list(range(gpu*24,(gpu+1)*24)))))
        files.append(path)
    validate_assignments(dict(assignment_files=[ref(p) for p in files]))
    value=json.loads(files[0].read_text());value['indices']=value['indices'][1:]
    files[0].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='complete predeclared'):
        validate_assignments(dict(assignment_files=[ref(p) for p in files]))


def test_shared_program_error_is_first_writer_wins(tmp_path):
    import json
    from tools.task_graph.p68_book_evaluate import signal_program_error
    signal_program_error(tmp_path,7,1);signal_program_error(tmp_path,28,2)
    assert json.loads((tmp_path/'STOP_PROGRAM_ERROR.json').read_text())==dict(index=7,returncode=1,automatic_retry=False)
