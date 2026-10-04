"""Process cleanup and static closure only; no scientific execution."""
import json
import os
from pathlib import Path
import sys
import pytest
from tools.task_graph import p53_pipeline as pipeline


def test_timeout_kills_detached_descendant_and_records_abort(tmp_path):
    pidfile=tmp_path/'child.pid'
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    with pytest.raises(RuntimeError,match='without retry'):
        pipeline.execute([sys.executable,'-c',code,str(pidfile)],os.environ.copy(),.5,tmp_path,'timeout')
    receipt=json.loads((tmp_path/'timeout.receipt.json').read_text())
    assert receipt['execution_status']=='engineering_timeout' and receipt['system_success'] is None
    assert (tmp_path/'timeout.ABORT.json').is_file()
    stat=Path('/proc')/pidfile.read_text()/'stat'
    assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z'


def test_nested_frozen_visual_import_in_source_closure():
    closure=pipeline.source_closure(pipeline.ROOT,['tools/task_graph/p53_pipeline.py'])
    assert 'prove/task_graph/p52_visual_features.py' in closure
