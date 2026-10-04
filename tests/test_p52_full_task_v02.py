from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from tools.task_graph import p52_full_task_v02 as runner


def test_three_arm_bindings_and_shared_joint():
    package=dict(schema='p52_stage_skill_v01',settings=dict(training=dict(ordinary=False)),data_sha256='same',config={},model={})
    def load(path,**kwargs):
        return dict(package,settings=dict(training=dict(ordinary=path.name=='bc')))
    with patch.object(runner.original,'prepare',return_value={'checkpoint':{},'sources':{}}),patch.object(runner.torch,'load',side_effect=load),patch.object(runner,'StagePolicy') as model,patch.object(runner,'Config'),patch.object(runner,'sha',side_effect=lambda p:str(p)),patch.object(runner,'source_closure',return_value=['new','old','sidecar']):
        model.return_value.state_dict.return_value={'weight':torch.ones(1)}
        b=runner.prepare(Path('binding'),Path('bc'),Path('joint'),Path('dino'),Path('lila'),'python','cuda')
    assert b['cells']==[dict(index=i,arm=a) for i,a in enumerate(('bc','fixed','online'))]
    assert b['checkpoints']['fixed']==b['checkpoints']['online']
    assert b['checkpoints']['bc']!=b['checkpoints']['fixed']
    assert set(b['sources'])=={'new','old','sidecar'}


def test_reject_old_scope():
    with pytest.raises(ValueError,match='scope'):
        runner.verify(dict(schema='p52.full-original-task.v1',cells=runner.CELLS))


def test_original_helpers_unchanged():
    assert runner.run_prefix is runner.original.run_prefix
    assert runner.run_insert is runner.original.run_insert
    assert runner.InsertJudge is runner.original.InsertJudge
