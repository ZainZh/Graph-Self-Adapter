from pathlib import Path
import pytest
from tools.task_graph import p70_queue as q


def fixture(tmp):
    freeze=tmp/'freeze.json';q.dump(freeze,{'frozen':True});review=tmp/'review.json';q.dump(review,dict(approved=True,freeze_sha256=q.sha(freeze)))
    jobs=[dict(id=f'c{i}',cwd=str(tmp),argv=['python','runner','--output','{output}'],freeze=dict(path=str(freeze),sha256=q.sha(freeze)),review=dict(path=str(review),sha256=q.sha(review))) for i in range(6)]
    m=tmp/'manifest.json';q.dump(m,dict(schema='prove.p70-queue.v1',jobs=jobs,gpu_ids=[0,1,2,3]))
    r=tmp/'root.json';q.dump(r,dict(approved=True,reviewer='root',manifest_sha256=q.sha(m),queue_sha256=q.sha(q.__file__)))
    return m,r


def test_isolated_configs_and_finite_jobs(tmp_path):
    m,r=fixture(tmp_path);active=set();configs=[];calls=[]
    def spawn(cmd,**kw):
        gpu=kw['env']['CUDA_VISIBLE_DEVICES'];assert gpu not in active;active.add(gpu);configs.append(kw['env']['LIBERO_CONFIG_PATH']);calls.append(cmd)
        d=Path(cmd[-1]);d.mkdir();q.dump(d/'SEAL.json',dict(freeze_sha256=q.read(m)['jobs'][0]['freeze']['sha256'],files={}));q.dump(d/'RESULT.json',dict(seal_sha256=q.sha(d/'SEAL.json')))
        class P:
            def poll(self):active.remove(gpu);return 0
        return P()
    result=q.run(m,r,tmp_path/'out',popen=spawn,sleep=lambda _:None)
    assert result['all_complete'] and len(calls)==6 and len(set(configs))==6


def test_error_stops_submission(tmp_path):
    m,r=fixture(tmp_path);calls=[]
    def spawn(cmd,**kw):
        calls.append(cmd)
        class P:
            def poll(self):return 1
        return P()
    result=q.run(m,r,tmp_path/'out',popen=spawn,sleep=lambda _:None)
    assert len(calls)==4 and len(result['unsubmitted'])==2 and not result['all_complete']


def test_changed_freeze_rejected_before_launch(tmp_path):
    m,r=fixture(tmp_path);(tmp_path/'freeze.json').write_text('{}')
    with pytest.raises(ValueError,match='Changed freeze'):q.run(m,r,tmp_path/'out')
    assert not (tmp_path/'out').exists()
