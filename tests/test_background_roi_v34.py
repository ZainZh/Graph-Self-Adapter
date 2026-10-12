import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
import api_grounding_v34 as api
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import drift_guard
from task_stream_view_v35 import ClearStateView

def test_background_roi_not_an_entity_but_real_entities_checked(tmp_path,monkeypatch):
    import prove.xtrainer.perception as perception
    camera=SimpleNamespace(capture=lambda:(1.,np.zeros((20,20,3),dtype=np.uint8),np.ones((20,20)),np.eye(3)))
    vision=SimpleNamespace(tracker=None,camera=camera,config={'role_boxes':{},'prompt_frame_output':str(tmp_path/'INITIAL_RGB.png'),'prompt_reply_path':str(tmp_path/'INITIAL_BOXES.json')})
    original=lambda self:SimpleNamespace(source='SAM')
    monkeypatch.setattr(perception.LiveVision,'capture',original)
    monkeypatch.setattr(api,'proposal',lambda *a,**k:{'frame_sha256':'x','role_boxes':{'apple':[1,1,5,5],'bowl':[6,6,9,9],'table_patch':[10,10,14,14]}})
    entry=SimpleNamespace(entry=SimpleNamespace(base=SimpleNamespace(SOURCES=()),backend=SimpleNamespace(main=lambda *a:None)))
    api.install('collect_apple',entry);perception.LiveVision.capture(vision)
    assert set(vision.config['role_boxes'])=={'apple','bowl'}
    assert vision.config['background_table_roi']==[10,10,14,14]
    sample=SimpleNamespace(stamp=1.,q=np.zeros(6))
    before=([sample],SimpleNamespace(stamp=1.,roles={'apple':np.zeros(3),'bowl':np.zeros(3)}))
    after=([sample],SimpleNamespace(stamp=1.,roles={'apple':None,'bowl':np.zeros(3)}))
    with pytest.raises(SafetyError,match='apple'):drift_guard(before,after,max_age_sec=.25,max_q_drift_rad=.015,max_role_drift_m=.01,now=1.)

def test_failed_seal_cannot_look_like_task_completion(tmp_path):
    (tmp_path/'PREDICTIONS_SEALED.json').write_text(json.dumps({'predictions':{'failure':'apple role lost','graph_completed':False,'graph_stages':8,'stage_trace':[{'completed':True}]*3}}))
    view=ClearStateView('collect_apple',0);view.attach(tmp_path);s=view.status()
    assert s['execution_state']=='failed';assert s['completed_nodes']==3;assert s['total_nodes']==8
    assert s['stop_reason']=='apple role lost'
