import importlib.util,json,sys,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/deployment'))
from task_live_view_v27 import TaskLiveView

def test_waiting_and_public_frame_http(tmp_path):
    v=TaskLiveView('collect_apple',0);url=v.start()
    try:
        with urllib.request.urlopen(url+'status.json') as r:
            d=json.loads(r.read());assert r.status==200;assert not d['frame_available']
        with urllib.request.urlopen(url+'frame.jpg') as r:assert r.status==204
        run=tmp_path/'run';run.mkdir();v.attach(run)
        Image.fromarray(np.full((20,30,3),120,dtype=np.uint8)).save(run/'INITIAL_RGB.png')
        (run/'NODE_STATUS.json').write_text(json.dumps({'stage_id':'approach'}))
        v.refresh()
        with urllib.request.urlopen(url+'status.json') as r:
            d=json.loads(r.read());assert d['frame_available'];assert d['node']['stage_id']=='approach';assert d['extra_camera_captures']==0
        with urllib.request.urlopen(url+'frame.jpg') as r:assert r.status==200;assert r.headers['Content-Type']=='image/jpeg'
        assert not (run/'RESULT.json').exists()
    finally:v.close()

def test_private_trace_is_never_a_frame(tmp_path):
    (tmp_path/'STATE_TRACE.npz').write_bytes(b'private')
    v=TaskLiveView('collect_cubes',0);v.attach(tmp_path);v.refresh()
    assert not v.status()['frame_available'];assert v.status()['frame_source'] is None

def test_initial_grounding_wait_is_visible(tmp_path):
    q=tmp_path/'ground/REQUEST.json';q.parent.mkdir()
    q.write_text(json.dumps({'reply_path':str(tmp_path/'not_written.json')}))
    v=TaskLiveView('put_mug_on_box',0);v.attach(tmp_path)
    assert 'Agent' in v.status()['message']

def test_finished_elapsed_stops_and_waiting_is_not_execution(tmp_path):
    import os
    now=time.time()
    frame=tmp_path/'INITIAL_RGB.png'
    Image.fromarray(np.zeros((10,10,3),dtype=np.uint8)).save(frame)
    os.utime(frame,(now-60,now-60))
    v=TaskLiveView('collect_apple',0);v.attach(tmp_path);v.refresh()
    s=v.status();assert s['execution_state']=='waiting';assert 59<=s['elapsed_wall_sec']<=62
    seal=tmp_path/'PREDICTIONS_SEALED.json';seal.write_text('{}');os.utime(seal,(now-5,now-5))
    s=v.status();assert s['execution_state']=='finished';assert 54<=s['elapsed_wall_sec']<=56
    assert v.status()['elapsed_wall_sec']==s['elapsed_wall_sec']
