import io,sys,json,threading,urllib.request,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from sim_preview_v33 import copy_graphics_snapshot
from task_stream_view_v33 import StreamingTaskView,STREAM_PAGE

def test_display_snapshot_isolated_from_live_physics():
    d=SimpleNamespace(qpos=np.array([1.,2.]),qvel=np.array([3.,4.]),mocap_pos=np.ones((1,3)),mocap_quat=np.ones((1,4)),time=8.)
    snap=SimpleNamespace(qpos=np.zeros(2),qvel=np.zeros(2),mocap_pos=np.zeros((1,3)),mocap_quat=np.zeros((1,4)),time=0.)
    robot=SimpleNamespace(lock=threading.RLock(),world=SimpleNamespace(data=d))
    m=copy_graphics_snapshot(robot,snap);assert m['physical_time_sec']==8.
    snap.qpos[:]=99;assert d.qpos.tolist()==[1.,2.];assert d.time==8.

def test_stream_delivers_multiple_new_jpegs_without_camera_reads():
    v=StreamingTaskView('collect_apple',0);url=v.start();stop=threading.Event()
    def publish():
        i=0
        while not stop.wait(.05):
            b=io.BytesIO();Image.fromarray(np.full((5,5,3),i%255,dtype=np.uint8)).save(b,format='JPEG');v.publish(b.getvalue(),{'captured_wall_sec':time.time(),'physical_time_sec':i/20});i+=1
    worker=threading.Thread(target=publish,daemon=True);worker.start()
    try:
        with urllib.request.urlopen(url+'stream.mjpg',timeout=2) as r:
            assert 'multipart/x-mixed-replace' in r.headers['Content-Type'];parts=[]
            for _ in range(3):
                assert r.readline().strip()==b'--frame';headers={}
                while True:
                    line=r.readline()
                    if line==b'\r\n':break
                    key,val=line.decode().split(':',1);headers[key.lower()]=val
                parts.append(r.read(int(headers['content-length'])));r.read(2)
            assert len(set(parts))>1
        assert v.status()['extra_camera_captures']==0
        assert '/stream.mjpg' in STREAM_PAGE
    finally:stop.set();worker.join();v.close()
