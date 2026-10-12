import sys,json,socket
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from ordinary_real_portable_v21 import check,load_bundle
import ordinary_real_entry_v21 as entry
from prove.xtrainer.io import SafetyError

@pytest.mark.parametrize('task,nodes',[('collect_apple',8),('collect_cubes',40),('put_knife_board_new',8),('sort_two_spoon',16)])
def test_task_bound_prepared_model_graph_reference_no_legacy_promotion(task,nodes):
    root=ROOT/'evaluations/multitask_parallel_v22/ordinary'/task/'portable_prepared_r1';bundle,report=load_bundle(root,task)
    assert report['package_verified'] and not report['promotion_verified'] and len(bundle['graph']['stages'])==nodes
    assert bundle['task']==task and bundle['skill'].model.input_dim==29
    assert not any(p.suffix=='.key' or p.name=='scene.xml' for p in root.rglob('*'))
    with pytest.raises(SafetyError,match='exact task'):check(root,'another_task')


def test_default_check_and_unqualified_shadow_execute_hold_before_device_or_key(monkeypatch):
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('connected'))
    original=Path.read_text
    def read(self,*a,**k):
        if self.suffix=='.key':pytest.fail('key read')
        return original(self,*a,**k)
    monkeypatch.setattr(Path,'read_text',read)
    root=ROOT/'evaluations/multitask_parallel_v22/ordinary/collect_apple/portable_prepared_r1'
    report=entry.main(['--task','collect_apple','check','--package',str(root)])
    assert not report['promotion_verified'] and report['devices_opened']==0
    for mode in ('shadow','execute'):
        with pytest.raises(SafetyError,match='receipt not present'):entry.main(['--task','collect_apple',mode,'--package',str(root)])
