import importlib.util
from pathlib import Path
import sys,json,io,urllib.error
import pytest
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_compact_judge_test',ROOT/'tools/deployment/compact_stage_judge.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

@pytest.fixture
def request_path(tmp_path):
    frames=[]
    for i in range(3):
        path=tmp_path/f'RGB_{i}.png';Image.new('RGB',(640,360),(i*50,0,0)).save(path)
        frames.append({'frame_id':f'f{i}','path':path.name,'sha256':m.digest(path),'observed_monotonic_sec':10.+i*.2,'public_robot':[{'TCP_m':[i*.003,0,0],'q':[999]*6},{'TCP_m':[0,0,0]}]})
    value={'stage_id':'c03.Grasp','stage_contract':{'stage_id':'c03.Grasp','phase':'Grasp','actor':'left_gripper','arm':'left','object':'white_mug','object_part':'visible_handle','target':'table','goal_relation':'tentative_holding','reference':{'SECRET_REFERENCE':'do not send'},'graph':'do not send'},'requested_facts':['holding','released'],'frames':frames,'public_robot':[{'actual_aperture_m':None,'q':[999]*6,'qdot':[999]*6,'gripperOpenClose':'close'},{'actual_aperture_m':.074}],'graph_sha256':'local_only','request_id':'local_id','Root_label':'secret_label','evaluator':'secret_private','secret':'do not send'}
    path=tmp_path/'REQUEST.json';path.write_text(json.dumps(value));return path


def test_context_whitelist_unknown_jaw_first_latest_and_original_timestamps(request_path):
    compact=m.compact_context(request_path);text=json.dumps(compact.context)
    assert [p.name for p in compact.images]==['RGB_0.png','RGB_2.png']
    assert compact.context['actual_actor_jaw_m'] is None and compact.context['first_last_public_TCP_delta_norm_m']==.006
    assert compact.context['frames'][1]['observed_monotonic_sec']==10.4
    for forbidden in ('SECRET_REFERENCE','secret_label','secret_private','qdot','999','local_only','local_id',str(request_path.parent)):assert forbidden not in text
    assert len(m.SYSTEM_PROMPT.split())<350

@pytest.mark.parametrize('failure',['hash','escape','time'])
def test_bad_frame_binding_rejected(request_path,failure):
    value=json.loads(request_path.read_text())
    if failure=='hash':value['frames'][2]['sha256']='bad'
    if failure=='escape':value['frames'][2]['path']='../outside.png'
    if failure=='time':value['frames'][2]['observed_monotonic_sec']=9.
    request_path.write_text(json.dumps(value))
    with pytest.raises(m.JudgeError):m.compact_context(request_path)


def test_unknown_overrides_true_missing_facts_unknown_and_extra_facts_rejected(request_path):
    compact=m.compact_context(request_path);value={'decision':'unknown','facts':{'holding':True},'evidence_frame_ids':['f0','f2'],'reason':'occluded'}
    assert m.parse_reply(value,compact)['facts']=={'holding':None,'released':None}
    value['decision']='continue';assert m.parse_reply(value,compact)['facts']['released'] is None
    value['facts']['success']=True
    with pytest.raises(m.JudgeError):m.parse_reply(value,compact)
    value['facts']={};value['evidence_frame_ids']=['f1']
    with pytest.raises(m.JudgeError):m.parse_reply(value,compact)


def test_fake_transport_low_reasoning_budget_no_secrets_or_path_and_one_call(request_path,tmp_path):
    compact=m.compact_context(request_path);key=tmp_path/'fake_key.txt';key.write_text('fake-key-only');calls=[]
    def opener(req,timeout):
        calls.append(req);payload=json.loads(req.data)
        assert timeout==30 and payload['max_tokens']==512 and payload['temperature']==0
        assert payload['reasoning']=={'effort':'low','exclude':True} and payload['provider']=={'require_parameters':True}
        assert len(payload['messages'][1]['content'])==3
        assert 'fake-key-only' not in req.data.decode() and str(request_path.parent) not in req.data.decode()
        reply={'decision':'continue','facts':{'holding':True},'evidence_frame_ids':['f0','f2'],'reason':'tentative supported grip'}
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':json.dumps(reply)}}],'usage':{'prompt_tokens':200,'completion_tokens':50,'total_tokens':250,'completion_tokens_details':{'reasoning_tokens':20,'SECRET':'never log'},'cost':.0001}}).encode())
    judge=m.OpenRouterJudge(key,maximum_calls=1,opener=opener);reply,record=judge.ask(compact)
    assert reply['facts']['released'] is None and record['usage']['completion_tokens_details']=={'reasoning_tokens':20.}
    assert 'fake-key-only' not in json.dumps(record) and 'SECRET' not in json.dumps(record)
    with pytest.raises(m.JudgeError,match='budget'):judge.ask(compact)
    assert len(calls)==1


def test_http_error_redacted_no_retry(request_path,tmp_path):
    key=tmp_path/'fake_key.txt';key.write_text('fake-key-only');calls=[]
    def opener(req,timeout):
        calls.append(req);raise urllib.error.HTTPError(m.ENDPOINT,400,'bad',{},io.BytesIO(b'Bearer fake-key-only data:image/png;base64,AAAAAAA'))
    judge=m.OpenRouterJudge(key,maximum_calls=1,opener=opener)
    with pytest.raises(m.JudgeError) as error:judge.ask(m.compact_context(request_path))
    assert 'fake-key-only' not in str(error.value) and 'AAAAAAA' not in str(error.value) and '400' in str(error.value)
    assert len(calls)==1

def test_invalid_json_still_records_billed_reasoning_and_cost(request_path,tmp_path):
    key=tmp_path/'fake_key.txt';key.write_text('fake-key-only')
    def opener(req,timeout):return io.BytesIO(json.dumps({'choices':[{'message':{'content':'invalid json'}}],'usage':{'total_tokens':250,'completion_tokens_details':{'reasoning_tokens':200},'cost':.002}}).encode())
    judge=m.OpenRouterJudge(key,maximum_calls=1,opener=opener)
    with pytest.raises(m.JudgeError) as error:judge.ask(m.compact_context(request_path))
    assert error.value.metrics['cost']==.002 and judge.records[0]['usage']['completion_tokens_details']['reasoning_tokens']==200
    assert 'fake-key-only' not in json.dumps(judge.records)

def test_positive_causal_facts_require_two_supplied_frames_and_real_time_span(request_path):
    compact=m.compact_context(request_path)
    for fact in ('holding','released','placement_complete'):
        compact.context['requested_facts']=[fact]
        value={'decision':'continue','facts':{fact:True},'evidence_frame_ids':['f2'],'reason':'looks stable'}
        with pytest.raises(m.JudgeError,match='f0\\+f2'):m.parse_reply(value,compact)
        value['evidence_frame_ids']=['f0','f2'];assert m.parse_reply(value,compact)['facts'][fact] is True
        saved=compact.context['frames'][1]['observed_monotonic_sec'];compact.context['frames'][1]['observed_monotonic_sec']=10.1
        with pytest.raises(m.JudgeError,match='.19sec'):m.parse_reply(value,compact)
        compact.context['frames'][1]['observed_monotonic_sec']=saved


def test_complete_missing_fact_unknown_and_false_fact_continue(request_path):
    compact=m.compact_context(request_path)
    assert 'supported-tentative' in compact.context['holding_contract'] and compact.context['judge_scope']=='semantic facts only; controller checks pose/progress separately'
    value={'decision':'complete','facts':{'holding':True},'evidence_frame_ids':['f0','f2'],'reason':'partial visible evidence'}
    reply=m.parse_reply(value,compact)
    assert reply['decision']=='unknown' and reply['facts']=={'holding':None,'released':None}
    value['facts']={'holding':True,'released':False};reply=m.parse_reply(value,compact)
    assert reply['decision']=='continue' and reply['facts']['released'] is False

def test_phase_specific_context_omits_inherited_future_goal_metadata(request_path):
    value=json.loads(request_path.read_text());value['stage_contract']['goal_relation']='accessible_to_right_arm'
    request_path.write_text(json.dumps(value));grasp=m.compact_context(request_path).context
    assert 'target' not in grasp and 'goal_relation' not in grasp and 'query' in grasp
    assert grasp['object']=='white_mug' and grasp['part']=='visible_handle' and 'holding_contract' in grasp
    value['stage_contract']['phase']='Complete';value['stage_contract']['target']='box';request_path.write_text(json.dumps(value))
    complete=m.compact_context(request_path).context
    assert complete['target']=='box' and complete['query_goal_relation']=='released_and_stable_on_declared_target'
    assert not {'part','holding_contract','goal_relation'} & set(complete)
    assert 'accessible_to_right_arm' not in json.dumps(grasp) and 'accessible_to_right_arm' not in json.dumps(complete)
