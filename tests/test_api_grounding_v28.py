import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
import api_grounding_v28 as api
from prove.xtrainer.io import SafetyError

def test_pixel_boxes_unknown_and_missing_roles():
    valid={'status':'proposal','role_boxes':{'apple':[1,2,9,10]},'role_points':{},'reason':'visible'}
    assert api.validate(valid,{'apple':'fruit'},20,20) is valid
    with pytest.raises(SafetyError):api.validate({**valid,'status':'unknown'},{'apple':'fruit'},20,20)
    with pytest.raises(SafetyError):api.validate(valid,{'bowl':'bowl'},20,20)
    with pytest.raises(SafetyError):api.validate({**valid,'role_boxes':{'apple':[0,0,1000,1000]}},{'apple':'fruit'},20,20)

def test_paid_transport_is_bound_and_identity_not_root(tmp_path,monkeypatch):
    key=tmp_path/'key';key.write_text('test-secret');monkeypatch.setattr(api,'KEY',key)
    f=tmp_path/'rgb.png';Image.fromarray(np.zeros((20,20,3),dtype=np.uint8)).save(f)
    def opener(req,timeout):
        d=json.loads(req.data);assert d['model']=='openai/gpt-6-astra';assert timeout==45
        value={'status':'proposal','role_boxes':{'apple':[1,2,9,10]},'role_points':{},'reason':'visible'}
        raw=json.dumps({'choices':[{'message':{'content':json.dumps(value)}}]}).encode()
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return raw
        return Response()
    p=api.proposal(f,{'apple':'fruit'},tmp_path/'evidence',opener=opener)
    assert p['grounding_source']=='openrouter/openai/gpt-6-astra'
    assert 'test-secret' not in (tmp_path/'evidence/REQUEST.json').read_text()
    assert p['frame_sha256']

def test_auto_initialization_reuses_only_retained_sample_then_refreshes(tmp_path,monkeypatch):
    import prove.xtrainer.perception as perception
    class Camera:
        def __init__(self):self.count=0
        def capture(self):
            self.count+=1;return (self.count,np.zeros((20,20,3),dtype=np.uint8),np.ones((20,20)),np.eye(3))
    cam=Camera();vision=SimpleNamespace(tracker=None,camera=cam,config={'role_boxes':{},'prompt_frame_output':str(tmp_path/'INITIAL_RGB.png'),'prompt_reply_path':str(tmp_path/'INITIAL_BOXES.json')})
    def original(self):
        first=self.camera.capture();second=self.camera.capture();assert first[0]==1 and second[0]==2
        assert self.config['prompt_frame_sha256'];return SimpleNamespace(source='SAM')
    monkeypatch.setattr(perception.LiveVision,'capture',original)
    def proposal(frame,roles,evidence):return {'frame_sha256':api.hashlib.sha256(frame.read_bytes()).hexdigest(),'role_boxes':{'apple':[1,2,9,10]},'role_points':{}}
    monkeypatch.setattr(api,'proposal',proposal)
    backend=SimpleNamespace(main=lambda x:None);entry=SimpleNamespace(entry=SimpleNamespace(base=SimpleNamespace(SOURCES=()),backend=backend))
    api.install('collect_apple',entry);monkeypatch.setattr(perception.LiveVision,'capture',perception.LiveVision.capture)
    obs=perception.LiveVision.capture(vision);assert vision.camera is cam;assert cam.count==2;assert 'openrouter/' in obs.source
