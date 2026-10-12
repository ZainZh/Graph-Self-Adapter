import sys,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from api_stage_judge_v30 import APIStageJudge,RealRootJudge
from prove.xtrainer.perception import Observation
from prove.xtrainer.io import SafetyError

def test_lift_receives_real_past_frame_and_current_stamp_unchanged(tmp_path,monkeypatch):
    frame=tmp_path/'PRE_LIFT_PUBLIC_RGB.png';Image.fromarray(np.zeros((10,10,3),dtype=np.uint8)).save(frame)
    prior={'RGB':str(frame),'RGB_sha256':hashlib.sha256(frame.read_bytes()).hexdigest(),'stamp':1.,'public_robot':[{'TCP_m':[0,0,0]}]}
    (tmp_path/'PRE_LIFT_PUBLIC_OBSERVATION.json').write_text(json.dumps(prior))
    rgb=np.ones((10,10,3),dtype=np.uint8);obs=Observation(5.,rgb,{'apple':np.zeros(3)},'public',np.eye(3),history=[(4.7,rgb,[]),(4.8,rgb,[]),(5.,rgb,[])])
    captured={}
    def ask(self,stage,facts,**kwargs):captured['before']=kwargs['before'];return ({'holding':None},kwargs['before'])
    monkeypatch.setattr(RealRootJudge,'ask',ask)
    judge=APIStageJudge(tmp_path/'REAL_AGENT_HANDOFFS',task='collect_apple',graph_sha='x',identity='ignored')
    judge.ask({'phase':'Lift'},['holding'],before=([],obs),recapture=lambda:([],obs))
    sent=captured['before'][1]
    assert [f[0] for f in sent.history]==[1.,4.8,5.]
    assert sent.stamp==obs.stamp==5.;assert obs.history[0][0]==4.7
    assert not sent.history[0][1].any()
    prior['RGB_sha256']='wrong';(tmp_path/'PRE_LIFT_PUBLIC_OBSERVATION.json').write_text(json.dumps(prior))
    with pytest.raises(SafetyError,match='binding changed'):
        judge.ask({'phase':'Lift'},['holding'],before=([],obs),recapture=lambda:([],obs))

def test_strict_schema_excludes_nonprotocol_pass():
    from compact_stage_judge import response_schema
    from types import SimpleNamespace
    spec=response_schema(SimpleNamespace(context={'requested_facts':['holding'],'frames':[{'frame_id':'f0'},{'frame_id':'f2'}]}))
    assert spec['type']=='json_schema';assert spec['json_schema']['strict'] is True
    props=spec['json_schema']['schema']['properties']
    assert 'pass' not in props['decision']['enum']
    assert 'complete' in props['decision']['enum']
    assert props['facts']['additionalProperties'] is False
