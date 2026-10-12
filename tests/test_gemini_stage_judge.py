"""No network/key reads: exercise the actual frozen bridge and API-schema parser."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from gemini_stage_judge import GeminiStageJudge,MODEL_IDENTITY,validate_config
from compact_stage_judge import parse_reply,JudgeError
from prove.xtrainer.io import RobotSample,SafetyError
from test_stage_trajectory import Clock

class Client:
    def __init__(self,clock):self.clock=clock;self.calls=0;self.fail=False;self.delay=.02;self.seen=[];self.decision='continue'
    def ask(self,compact):
        self.calls+=1;self.seen.append(compact.context);self.clock.sleep(self.delay)
        if self.fail:raise JudgeError('fake provider error; no retry')
        value={'decision':self.decision,'facts':{fact:True for fact in compact.context['requested_facts']},'evidence_frame_ids':['f0','f2'],'reason':'tentative supported visual grip'}
        return parse_reply(value,compact),{'wall_sec':self.delay,'usage':{'completion_tokens_details':{'reasoning_tokens':3}},'cost':.0001,'call_count':self.calls}


def setup(tmp_path):
    clock=Clock();clock.t=10.;client=Client(clock)
    def capture(qchange=0):
        samples=[RobotSample(clock(),np.full(6,qchange),np.zeros(6),np.zeros(3),np.eye(3),None,'fake_own_actual') for _ in range(2)]
        rgb=np.zeros((360,640,3),dtype=np.uint8)
        history=[(clock()-.4,rgb,[{'TCP_m':[0,0,0]},{'TCP_m':[0,0,0]}]),(clock()-.2,rgb,[]),(clock(),rgb,[{'TCP_m':[.001,0,0]},{'TCP_m':[0,0,0]}])]
        return samples,SimpleNamespace(stamp=clock(),rgb=rgb,history=history,roles={'white_mug':np.zeros(3)})
    judge=GeminiStageJudge(tmp_path/'handoffs',task='put_mug_on_box',graph_sha='bound-graph',identity='codex-old-root-must-not-be-used',key_file='/never-read-fake-key',client=client,clock=clock,sleep=clock.sleep)
    stage={'stage_id':'c03.Grasp','phase':'Grasp','arm':'left','actor':'left_gripper','object':'white_mug','object_part':'handle','holding_contract':'supported','target':'future-box','goal_relation':'irrelevant-future','Root_label':'must_not_send','reference':{'q':[999]*6}}
    return clock,client,capture,judge,stage


def test_provider_identity_local_receipts_two_frames_and_no_labels(tmp_path):
    clock,client,capture,judge,stage=setup(tmp_path);before=capture();facts,after=judge.ask(stage,['holding'],before=before,recapture=capture)
    assert facts=={'holding':True} and judge.identity==MODEL_IDENTITY and client.calls==1
    assert judge.records[0]['backend']=='OpenRouter' and judge.records[0]['actual_root_identity']==MODEL_IDENTITY
    assert 'Root_label' not in json.dumps(client.seen) and '999' not in json.dumps(client.seen) and 'target' not in client.seen[0]
    folder=next((tmp_path/'handoffs').glob('*/REQUEST.json')).parent
    assert json.loads((folder/'API_ADAPTER_RECEIPT.json').read_text())['not_human_Root_approval'] is True
    assert json.loads((folder/'ROOT_APPROVAL.json').read_text())['actual_root_identity']==MODEL_IDENTITY
    assert judge.records[0]['metrics']['usage']['completion_tokens_details']['reasoning_tokens']==3


def test_api_latency_followed_by_fresh_recapture_and_real_pose_drift_invalidates(tmp_path):
    clock,client,capture,judge,stage=setup(tmp_path);client.delay=3.;before=capture();seen=[]
    def recapture():seen.append(clock());return capture(.02)
    facts,_=judge.ask(stage,['holding'],before=before,recapture=recapture)
    assert len(seen)==1 and seen[0]>=13. and facts=={'holding':None}
    assert 'drifted' in judge.records[0]['postreply_guard_failure']


def test_unknown_never_grants_holding_and_provider_failure_has_no_fallback(tmp_path):
    clock,client,capture,judge,stage=setup(tmp_path);client.decision='unknown'
    assert judge.ask(stage,['holding'],before=capture(),recapture=capture)[0]=={'holding':None}
    client.fail=True
    with pytest.raises(SafetyError,match='failed closed'):judge.ask(stage,['holding'],before=capture(),recapture=capture)
    assert client.calls==2 and judge.api_records[-1]['failed_closed'] and len(judge.records)==1


def test_six_calls_budget_and_unsupported_anomaly_stop_without_key_or_api(tmp_path):
    clock,client,capture,judge,stage=setup(tmp_path)
    for _ in range(6):judge.ask(stage,['holding'],before=capture(),recapture=capture)
    with pytest.raises(SafetyError,match='budget'):judge.ask(stage,['holding'],before=capture(),recapture=capture)
    assert client.calls==6
    stage['phase']='Lift'
    with pytest.raises(SafetyError,match='unsupported-phase'):judge.ask(stage,['holding'],before=capture(),recapture=capture)
    assert client.calls==6


def test_static_config_check_never_reads_key(tmp_path):
    validate_config({'backend':'OpenRouter','model':'google/gemini-3.8-flash','maximum_calls':6,'key_file':str(tmp_path/'nonexistent-key')})
    assert not (tmp_path/'nonexistent-key').exists()
